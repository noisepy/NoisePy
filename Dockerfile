ARG PYTHON_VERSION=3.10.12
FROM --platform=$TARGETPLATFORM conda/miniconda3

RUN conda create -y --name noisepy python=${PYTHON_VERSION} numcodecs && conda init bash && source /root/.bashrc && conda activate noisepy

ARG VERSION
RUN pip3 install noisepy-seis==${VERSION}
ENTRYPOINT ["noisepy"]
CMD ["--help"]
