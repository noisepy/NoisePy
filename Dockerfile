FROM --platform=$TARGETPLATFORM ubuntu:22.04
ARG PYTHON_VERSION=3.10.12

# Avoid timezone prompts during python installation
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y python3.10 && \
    apt-get install -y python3-pip

ARG VERSION
RUN pip3 install noisepy-seis==${VERSION}
ENTRYPOINT ["noisepy"]
CMD ["--help"]
