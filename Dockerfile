FROM --platform=$BUILDPLATFORM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y libopenmpi-dev && \
    apt-get install -y python3 && \
    apt-get install -y python3-pip && \
    pip3 install --upgrade pip

# Copy and install requirements first so this layer doesn't change when copy the local files below
COPY requirements.txt .
RUN pip3 install -r requirements.txt
ADD . .
WORKDIR /src
ENTRYPOINT ["python3", "noisepy.py"]
CMD ["--help"]
