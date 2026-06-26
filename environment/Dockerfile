# hash:sha256:ded5e82a9e2d96c1984231f24034a755b34a66dab487f9b6d41c9c8f50bf13e2
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:v5.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    h5py==3.16.0 \
    hdmf==4.3.1 \
    hedtools==1.1.0 \
    matplotlib==3.11.0 \
    ndx-events==0.4.0 \
    ndx-hed==0.2.0 \
    numpy==2.2.6 \
    pandas==2.3.3 \
    pynwb==3.1.3 \
    scipy==1.15.3 \
    six==1.17.0
