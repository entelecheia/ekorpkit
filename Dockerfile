FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y fontconfig fonts-nanum curl git
# for disco-diffusion
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y tzdata imagemagick ffmpeg \ 
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up environment variables
ENV WORK_DIR=/tests
ENV EKORPKIT_CONFIG_DIR=$WORK_DIR/ekorpkit/tests/config
ENV EKORPKIT_WORKSPACE_ROOT=/workspace
ENV EKORPKIT_PROJECT_NAME=ekorpkit-test
ENV KMP_DUPLICATE_LIB_OK TRUE
ENV PIP_DEFAULT_TIMEOUT 100
ENV DS_BUILD_FUSED_ADAM 1
# for disco-diffusion
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV KMP_DUPLICATE_LIB_OK TRUE

WORKDIR $WORK_DIR

RUN pip install --upgrade --no-cache-dir pip && \
    pip install --no-cache-dir \
        "ekorpkit[all]" wandb transformers simpletransformers hydra-core hydra-colorlog \
        imageio "pyspng==0.1.0" lpips timm "pytorch-lightning>=1.0.8" torch-fidelity \
        einops ftfy seaborn flax unidecode "opencv-python==4.5.5.64"

COPY ./scripts/tests ./scripts

RUN curl -Os https://uploader.codecov.io/latest/linux/codecov
RUN chmod +x codecov

CMD ["bash"]
