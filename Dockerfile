FROM --platform=$BUILDPLATFORM ubuntu:20.04

# Avoid timezone prompts during python installation
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y libopenmpi-dev && \
    apt-get install -y python3.8 && \
    apt-get install -y python3-pip && \
    pip3 install --upgrade pip

ARG VERSION
RUN pip3 install noisepy-seis==${VERSION}
ENTRYPOINT ["noisepy"]
CMD ["--help"]
