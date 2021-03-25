#!/bin/bash -eux

image="us-docker.pkg.dev/replicate/example-repos/rife"
docker build --tag "$image" -f docker/Dockerfile .
docker push "$image"
