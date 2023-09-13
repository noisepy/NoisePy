#!/bin/bash
# Script to publish a docker container from the current state of the repo.
# Useful for iterating on AWS
rm ./dist/*.whl
hatch build -t wheel
python script/gen_req.py dist/*.whl > ./dist/_reqs.txt

docker buildx build -f Dockerfile.dev --platform=linux/amd64,linux/arm64 -t ghcr.io/uw-ssec/dev-noisepy --push .
# docker buildx build -f script/write_speed/Dockerfile --platform=linux/amd64,linux/arm64 -t ghcr.io/uw-ssec/dev-noisepy --push .
