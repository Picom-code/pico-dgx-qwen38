#!/bin/bash
# casc_run.sh [MAXWIN] : run casc.py on the GPU in NVIDIA's PyTorch container (inside a lease: it uses the GPU). Container name q38-study-A.
M=$(tr '\n' ',' < $HOME/q38/runs/C10_marks.txt | sed 's/,$//')
docker run --rm --name q38-study-A --gpus all --ipc=host -v $HOME/q38/hf:/hf:ro -v $HOME/q38/prof:/p:ro -v $HOME/q38/kit-A10:/k:ro -e CASC_SET=$CASC_SET --entrypoint python3 nvcr.io/nvidia/pytorch:26.07-py3 /k/casc.py /hf/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/b9ce18a6887e18c089fdea8509a33e63429a3300 /p/c10_fit /p/c10_test ${1:-3000} "$M" 2>&1 | grep -v '^$' | tail -60
