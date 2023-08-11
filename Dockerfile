FROM --platform=$TARGETPLATFORM ubuntu:22.04

# Avoid timezone prompts during python installation
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y libopenmpi-dev && \
    apt install -y python3.10 && \
    apt install -y python3-pip && \
    pip3 install --upgrade pip

ARG VERSION
RUN pip3 install noisepy-seis==${VERSION}
ENTRYPOINT ["noisepy"]
CMD ["--help"]
