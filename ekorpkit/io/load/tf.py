import functools
import tensorflow as tf

# import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds


def dumping_dataset(split, shuffle_files=False):
    del shuffle_files
    if split == "train":
        ds = tf.data.TextLineDataset(
            [
                "gs://scifive/finetune/BC5CDR-chem/train.tsv_cleaned.tsv",
            ]
        )
    else:
        ds = tf.data.TextLineDataset(
            [
                "gs://scifive/finetune/BC5CDR-chem/test.tsv_cleaned.tsv",
            ]
        )
    # Split each "<t1>\t<t2>" example into (input), target) tuple.
    ds = ds.map(
        functools.partial(
            tf.io.decode_csv,
            record_defaults=["", ""],
            field_delim="\t",
            use_quote_delim=False,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    # Map each tuple to a {"input": ... "target": ...} dict.
    ds = ds.map(lambda *ex: dict(zip(["input", "target"], ex)))
    return ds


def ner_preprocessor(ds):
    def normalize_text(text):
        """Lowercase and remove quotes from a TensorFlow string."""
        return text

    def to_inputs_and_targets(ex):
        """Map {"inputs": ..., "targets": ...}->{"inputs": ner..., "targets": ...}."""
        return {
            "inputs": tf.strings.join(
                ["bc5cdr_chem_ner: ", normalize_text(ex["input"])]
            ),
            "targets": normalize_text(ex["target"]),
        }

    return ds.map(
        to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )


print("A few raw validation examples...")
for ex in tfds.as_numpy(dumping_dataset("train").take(5)):
    print(ex)

ds = dumping_dataset("train")
df = tfds.as_dataframe(ds)
df.tail()


# t5.data.TaskRegistry.remove('ncbi_ner')
# t5.data.TaskRegistry.add(
#     "ncbi_ner",
#     # Supply a function which returns a tf.data.Dataset.
#     dataset_fn=dumping_dataset,
#     splits=["train", "validation"],
#     # Supply a function which preprocesses text from the tf.data.Dataset.
#     text_preprocessor=[ner_preprocessor],
#     # Lowercase targets before computing metrics.
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     # We'll use accuracy as our evaluation metric.
#     metric_fns=[t5.evaluation.metrics.accuracy,
#                t5.evaluation.metrics.sequence_accuracy,
#                 ],
#     output_features=t5.data.Feature(vocabulary=t5.data.SentencePieceVocabulary(vocab)),
#     # output_features=t5.data.Feature(vocabulary=t5.data.SentencePieceVocabulary(vocab))
# )
