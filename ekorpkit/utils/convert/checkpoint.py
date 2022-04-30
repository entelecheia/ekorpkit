# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert ELECTRA checkpoint."""


import os
import glob
import logging
import orjson as json
from pathlib import Path

import torch

from transformers import (
    ElectraConfig,
    ElectraForMaskedLM,
    ElectraForPreTraining,
    load_tf_weights_in_electra,
)

log = logging.getLogger(__name__)


def convert_tf_checkpoint_to_pytorch(
    tf_checkpoint_path, config_file, pytorch_dump_path, model_type, load_weights_func
):
    # Initialise PyTorch model
    config = ElectraConfig.from_json_file(config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))

    if model_type == "discriminator":
        model = ElectraForPreTraining(config)
    elif model_type == "generator":
        model = ElectraForMaskedLM(config)
    else:
        raise ValueError(
            "The discriminator_or_generator argument should be either 'discriminator' or 'generator'"
        )

    # Load weights from tf checkpoint
    globals()[load_weights_func](
        model, config, tf_checkpoint_path, discriminator_or_generator=model_type
    )

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


def load_tf2_weights_in_electra(
    model, config, tf_checkpoint_path, discriminator_or_generator="discriminator"
):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        log.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    log.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    log.info("Loading names and arrays from TensorFlow checkpoint.\n")
    for name, shape in init_vars:

        m_names = name.split("/")
        if name == "_CHECKPOINTABLE_OBJECT_GRAPH" or m_names[0] in [
            "global_step",
            "temperature",
            "save_counter",
            "phase",
            "step",
        ]:
            log.info(f" - Skipping non-model layer {name}")
            continue
        if "optimizer" in name:
            log.info(f" - Skipping optimization layer {name}")
            continue

        array = tf.train.load_variable(tf_path, name)
        arrays.append(array)

        # convet layer/num to layer_num
        m = re.search(r"layer\/\d+", name)
        if m:
            name = name.replace(m.group(0), m.group(0).replace("/", "_"))

        # remove variable name ending: .ATTRIBUTES/VARIABLE_VALUE
        name = name.replace("/.ATTRIBUTES/VARIABLE_VALUE", "")

        log.info("Loading TF weight {} with shape {}".format(name, shape))
        names.append(name)

    log.info("Converting TensorFlow weights to PyTorch weights.\n")
    for name, array in zip(names, arrays):
        original_name: str = name
        try:
            if discriminator_or_generator == "discriminator":
                if name.startswith("model/generator/") or name.startswith("generator/"):
                    log.info(f"Skipping generator: {original_name}")
                    continue
            elif discriminator_or_generator == "generator":
                if name.startswith("model/discriminator/") and not name.startswith(
                    "model/discriminator/electra/embeddings"
                ):
                    log.info(f"Skipping discriminator: {original_name}")
                    continue
                elif name.startswith("electra/") and not name.startswith(
                    "electra/embeddings/"
                ):
                    log.info(f"Skipping discriminator: {original_name}")
                    continue
                elif name.startswith("generator/"):
                    name = name.replace("generator/", "electra/")
            # in case, tensorflow version is higher than 2.0.0, change the names
            if name.startswith("model/discriminator/") or name.startswith(
                "model/generator/"
            ):
                name = name.replace("model/discriminator/", "")
                name = name.replace("model/generator/", "")
                name = name.replace("bert_output", "output")
                name = name.replace("dense_output", "output")
                name = name.replace("self_attention", "self")
                name = name.replace(
                    "/embeddings/position_embeddings/embeddings",
                    "/embeddings/position_embeddings",
                )
                name = name.replace(
                    "/embeddings/token_type_embeddings/embeddings",
                    "/embeddings/token_type_embeddings",
                )
                name = name.replace("/embeddings/weight", "/embeddings/word_embeddings")

            name = name.replace("dense_1", "dense_prediction")
            name = name.replace(
                "generator_predictions/output_bias", "generator_lm_head/bias"
            )

            name = name.split("/")
            # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
            pointer = model
            # model_names = dir(model)
            # logger.info(model_names)
            for m_name in name:
                if m_name not in [
                    "model",
                    ".OPTIMIZER_SLOT",
                    "optimizer",
                    "base_optimizer",
                    "v",
                    ".ATTRIBUTES",
                    "VARIABLE_VALUE",
                ]:
                    # logger.info(f'processing {m_name} from {name}')
                    if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                        scope_names = re.split(r"_(\d+)", m_name)
                    else:
                        scope_names = [m_name]
                    if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                        pointer = getattr(pointer, "weight")
                    elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                        pointer = getattr(pointer, "bias")
                    elif scope_names[0] == "output_weights":
                        pointer = getattr(pointer, "weight")
                    elif scope_names[0] == "squad":
                        pointer = getattr(pointer, "classifier")
                    else:
                        pointer = getattr(pointer, scope_names[0])
                    if len(scope_names) >= 2:
                        num = int(scope_names[1])
                        pointer = pointer[num]
            if m_name.endswith("_embeddings"):
                pointer = getattr(pointer, "weight")
            elif m_name == "kernel":
                array = np.transpose(array)
            try:
                assert pointer.shape == array.shape, original_name
            except AssertionError as e:
                e.args += (pointer.shape, array.shape)
                raise
            print("Initialize PyTorch weight {}".format(name), original_name)
            pointer.data = torch.from_numpy(array)
        except AttributeError as e:
            print("Skipping {}".format(original_name), name, e)
            continue
    return model
