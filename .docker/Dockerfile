ARG BUILD_FROM="pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime"
FROM $BUILD_FROM

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt-get update \
    && apt-get install -y fontconfig fonts-nanum curl git tzdata locales imagemagick ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_DEFAULT_TIMEOUT 100
ENV DS_BUILD_FUSED_ADAM 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV KMP_DUPLICATE_LIB_OK TRUE
ENV EKORPKIT_WORKSPACE_ROOT=/workspace
WORKDIR $EKORPKIT_WORKSPACE_ROOT

ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL True
ARG PIP_PACKAGE_NAME=""
RUN pip install --upgrade --no-cache-dir pip && \
    pip install --no-cache-dir $PIP_PACKAGE_NAME

RUN rm -rf /workspace && mkdir -p /workspace

CMD ["bash"]
