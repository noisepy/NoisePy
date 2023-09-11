ARG PYTHON_VERSION=3.10.12
FROM --platform=$TARGETPLATFORM python:${PYTHON_VERSION}
# FROM --platform=$TARGETPLATFORM continuumio/miniconda3

# RUN conda create -y --name noisepy python=${PYTHON_VERSION} numcodecs && conda activate noisepy
# RUN conda install -c conda-forge obspy
# RUN conda init bash && . /root/.bashrc && conda activate noisepy

ARG VERSION
RUN pip3 install noisepy-seis==${VERSION}
ENTRYPOINT ["noisepy"]
CMD ["--help"]
