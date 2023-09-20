ARG PYTHON_VERSION=3.10.12
FROM --platform=$TARGETPLATFORM python:${PYTHON_VERSION}

ARG VERSION
RUN pip3 install "noisepy-seis[aws]==${VERSION}"
ENTRYPOINT ["noisepy"]
CMD ["--help"]
