FROM --platform=$BUILDPLATFORM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y libopenmpi-dev && \
    apt-get install -y python3 && \
    apt-get install -y python3-pip && \
    pip3 install --upgrade pip

RUN pip3 install noisepy-seis
ENTRYPOINT ["noisepy"]
CMD ["--help"]
