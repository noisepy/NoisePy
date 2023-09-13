ARG PYTHON_VERSION=3.10.12
FROM --platform=$TARGETPLATFORM python:${PYTHON_VERSION}

# This is a version of Dockerfile to build from a local wheel
# and run the write speed test
# Build with:
# script/docker_dev.sh

# Split out the install of dependencies and the noisepy code so
# the dependencies layer gets cached. Much faster build times this way.
COPY ./dist/_reqs.txt .
RUN pip install -r _reqs.txt
RUN pip install tiledb==0.22.3
COPY ./dist/*.whl .
RUN pip install --no-deps *.whl

COPY ./script/write_speed/write.py .
COPY ./script/write_speed/params.json .
ENTRYPOINT ["python"]
