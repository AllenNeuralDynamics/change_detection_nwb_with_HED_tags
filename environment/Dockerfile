# Code Ocean reproducible-run environment.
# Versions pinned to the stack the pipeline was validated against.
FROM python:3.10-slim

# System libs needed by h5py / hdf5 and general builds.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libhdf5-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (exact versions validated for this pipeline).
RUN pip install --no-cache-dir \
        numpy==2.2.6 \
        pandas==2.3.3 \
        scipy==1.15.3 \
        h5py==3.16.0 \
        six==1.17.0 \
        pynwb==3.1.3 \
        hdmf==4.3.1 \
        ndx-events==0.4.0 \
        ndx-hed==0.2.0 \
        hedtools==1.1.0 \
        jupyter==1.1.1
