FROM --platform=$TARGETPLATFORM python:3.10.12

ARG VERSION
RUN pip3 install noisepy-seis==${VERSION}
ENTRYPOINT ["noisepy"]
CMD ["--help"]
