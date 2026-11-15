#!/bin/bash
# mbrun.sh [BIN=mb] [ARGS]: run the microbench on the GPU (inside a test window only: DeepSeek must be stopped).
B=${1:-mb}; shift
docker run --rm --name q38-run-mb --gpus all -v $HOME/q38/mb:/mb --entrypoint /mb/$B nvcr.io/nvidia/pytorch:26.07-py3 "$@" 2>&1
