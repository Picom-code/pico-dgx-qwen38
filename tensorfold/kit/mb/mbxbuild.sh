#!/bin/bash
# mbxbuild.sh : compile the expert microbench twice (stock arithmetic: mbx; native E2M1/E4M3 decodes on sm_121a: mbxn). CPU-capped, no GPU.
D=$HOME/q38/mb
docker run --rm --name q38-mbbuild --cpus 6 --memory 8g -v $D:/mb -v $HOME/q38/build/tf/zig/kernels/cuda:/k:ro --entrypoint bash nvcr.io/nvidia/pytorch:26.07-py3 -c \
  "cd /mb && nvcc -O3 -std=c++17 -arch=sm_121 -I /k ${1:-mbx}.cu -o ${1:-mbx} 2> ${1:-mbx}.err; echo rc=\$?; grep -E 'error' ${1:-mbx}.err | head -20; nvcc -O3 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -DQ38_NATIVE_FP -I /k ${1:-mbx}.cu -o ${1:-mbx}n 2> ${1:-mbx}n.err; echo rcn=\$?; grep -E 'error' ${1:-mbx}n.err | head -20; chown $(id -u):$(id -g) ${1:-mbx}* 2>/dev/null; ls -la ${1:-mbx} ${1:-mbx}n 2>/dev/null"
