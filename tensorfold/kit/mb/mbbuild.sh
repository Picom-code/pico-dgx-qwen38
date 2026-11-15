#!/bin/bash
# mbbuild.sh [SRC=mb.cu] [OUT=mb]: compile the microbench in NVIDIA's PyTorch container (nvcc), CPU-capped, no GPU.
S=${1:-mb.cu}; O=${2:-mb}; D=$HOME/q38/mb
docker run --rm --name q38-mbbuild --cpus 6 --memory 8g -v $D:/mb -v $HOME/q38/build/tf/zig/kernels/cuda:/k:ro --entrypoint bash nvcr.io/nvidia/pytorch:26.07-py3 -c \
  "cd /mb && nvcc -O3 -std=c++17 -arch=sm_121 -I /k $S -o $O 2> $O.err; echo rc=\$?; grep -E 'error' $O.err | head -20; chown $(id -u):$(id -g) $O $O.err 2>/dev/null; ls -la $O 2>/dev/null"
