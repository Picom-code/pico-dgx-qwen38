#!/bin/bash
# eng.sh ENGINE [VARIANT] : start one Qwen engine on spark-923b as container q38-run-*, API on 127.0.0.1:18301 only.
# Engines: tf (MiaAI TensorFold 1.1.4 + 17 patches, official FP4-Local), fly (custom engine R custom engine R v0.3.13, EXL3 4.05 bpw),
#          flare (a third engine prepared gb10 core of FP4-Local), sal (another vLLM recipe vLLM recipe, NVFP4)
# eng.sh stop  removes every q38-run-* container. No --privileged anywhere; model dirs are mounted read-only.
E=$1; V=${2:-default}; Q=$HOME/q38; PORT=18301; T0=$(date +%s)
ready() { # ready PATH TIMEOUT NAME
  for i in $(seq $(( $2 / 2 ))); do
    curl -sf -m 3 http://127.0.0.1:$PORT$1 >/dev/null 2>&1 && { echo "READY $E/$V in $(( $(date +%s)-T0 )) s; MemAvailable $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GiB; GPU MiB $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | paste -sd+)"; return 0; }
    [ "$(docker inspect -f '{{.State.Running}}' $3 2>/dev/null)" = true ] || { echo "FAILED $E/$V: container exited"; docker logs --tail 40 $3 2>&1; return 1; }
    sleep 2
  done; echo "FAILED $E/$V: not ready after $2 s"; docker logs --tail 40 $3 2>&1; return 1
}
case $E in
stop) docker ps -a --format '{{.Names}}' | grep '^q38-run' | xargs -r docker rm -f; exit 0 ;;
tf)
  cd $Q/src/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
  export HF_CACHE=$Q/hf KERNEL_CACHE=$Q/tf/kcache STATE_DIR=$Q/tf/state LOG_DIR=$Q/tf/logs HOST=127.0.0.1 PORT=$PORT CONTAINER_NAME=q38-run-tf PREPARE=0 WAIT_TIMEOUT=900
  case $V in
    default) ;;                                   # README: 5 at once, 262,144 window, fp8 KV, vision, MTP drafts
    p16) export PARALLEL=16 TF_FLASHNEXT_PRODUCT_STREAMS=16 ;;   # obvious tuned variant: 16 slots
    p16bf16) export PARALLEL=16 TF_FLASHNEXT_PRODUCT_STREAMS=16 KV_DTYPE=bf16 ;;
    bf16) export KV_DTYPE=bf16 PARALLEL=3 ;;      # exact KV cache (3 full windows fit)
    nodraft) export DRAFTS=0 ;;
    *) echo "variant?"; exit 2 ;;
  esac
  ./start.sh > $Q/logs/tf_$V.start.log 2>&1; rc=$?; tail -12 $Q/logs/tf_$V.start.log | sed 's/\x1b\[[0-9;]*m//g'
  [ $rc = 0 ] && ready /v1/models 60 q38-run-tf ;;
fly)
  R=$Q/custom engine R-v0.3.13-linux-aarch64-cu130-sm121
  printf 'SPARK_HOME=/opt/custom engine R\nSPARK_PYTHON=python3\nSPARK_BIN='${FLYBIN:-/opt/custom engine R/bin/custom engine R}'\nSPARK_PORT=8888\nQWEN_MODEL=/models/exl3\nQWEN_ASSETS=/opt/custom engine R/assets/qwen38\n' > $Q/custom engine R/spark.env
  X=""; case $V in default) ;; nodraft) X="-e QWEN_SPEC_CUMCONF=0 -e QWEN_SPEC_K=0 -e QWEN_SPEC_KMAX=0" ;; nodraft2) X="-e QWEN_MTP=0 -e QWEN_SERVE_GRAPH=0" ;; ctx262k) X="-e QWEN_KV_TOKENS=262144" ;; *) echo "variant?"; exit 2 ;; esac
  docker run -d --name q38-run-fly --gpus all --ipc=host --ulimit memlock=-1 -p 127.0.0.1:$PORT:8888 $X $FLYMOUNT \
    -v $R:/opt/custom engine R:ro -v $Q/custom engine R/spark.env:/etc/spark.env:ro -e SPARK_ENV=/etc/spark.env \
    -v $(realpath ${FLYMODEL:-$Q/models/turboderp-exl3-4.05bpw}):/models/exl3:ro --entrypoint bash vllm/vllm-openai:nightly-aarch64 \
    -c 'bash /opt/custom engine R/engine-rs/serve/start.sh qwen; sleep 2; exec tail -n +1 -F /tmp/spark-qwen-engine.log /tmp/spark-qwen-front.log' >/dev/null
  ready /v1/models 600 q38-run-fly ;;
flare)
  F=ghcr.io/local-inference-lab/flare:release-972e28e-main-20261107-arm64
  X=""; case $V in default) ;; nodraft) X="--mtp-draft-tokens 0" ;; ram) X="--embedding-table-storage ram" ;; ramnodraft) X="--embedding-table-storage ram --mtp-draft-tokens 0" ;; *) echo "variant?"; exit 2 ;; esac
  # disk-backed n-gram rows use io_uring, which docker's default seccomp profile blocks: those variants run with seccomp=unconfined (not privileged)
  SEC="--security-opt seccomp=unconfined"; case $V in ram*) SEC="" ;; esac
  FP=$(python3 -c "import json,hashlib;print(hashlib.sha256(bytes(json.load(open('$Q/flare/publisher.json'))[0]['public_key'])).hexdigest())")
  MODELDIR=$(realpath ${FLAREMODEL:-$Q/models/flare-gb10}); mkdir -p $Q/flare/state; chmod 777 $Q/flare/state
  frun() { docker rm -f q38-run-flare >/dev/null 2>&1; docker run -d --name q38-run-flare --gpus 'device=0,capabilities=compute' $SEC --ulimit memlock=-1:-1 --stop-timeout 60 -p 127.0.0.1:$PORT:8000 \
    --mount type=bind,src=$MODELDIR,dst=/models,readonly --mount type=bind,src=$Q/flare,dst=/pub,readonly --mount type=bind,src=$Q/flare/state,dst=/state \
    $F run /models/model.safetensors --publisher-keys /pub/publisher.json --trust-store /state/trust.toml --dangerously-accept-publisher local-inference-lab --dangerously-approve-key $1 --gpus 0 --host 0.0.0.0 --port 8000 $X >/dev/null; }
  frun $FP; sleep 8
  if [ "$(docker inspect -f '{{.State.Running}}' q38-run-flare 2>/dev/null)" != true ]; then
    docker logs q38-run-flare 2>&1 | tail -8 | cut -c1-300; FP2=$(docker logs q38-run-flare 2>&1 | grep -oE '[0-9a-f]{64}' | grep -v $FP | head -1)
    [ -n "$FP2" ] && { echo "retry with fingerprint from the error text"; frun $FP2; }
  fi
  ready /v1/models 900 q38-run-flare ;;
tfb)  # TensorFold binary dir bind-mounted into an image that has the CUDA userland (for boxes without the MiaAI v1.1.4 image, and for stock releases)
  #   TFBIN (dir mounted at /opt/tensorfold), TFEXE (path inside), TFIMG, TFMODEL; variants: default bf16 p16 p16bf16 nodraft stock stockbf16
  BINDIR=${TFBIN:-$Q/tfbin/v114}; EXE=${TFEXE:-/opt/tensorfold/native/bin/tensorfold-native}; MODEL=$(realpath ${TFMODEL:-$Q/tmpmodels/nvidia-fp4-local})
  IMG=${TFIMG:-ghcr.io/miaai-lab/qwen3.8-flash-next-single-dgx-spark-tensorfold:zig-db28187-4144696f21e3}
  PAR=5; KV=fp8; DR=""; PS=5; MIA=1; CTX=262144
  case $V in default) ;; bf16) KV=bf16; PAR=3 ;; p16) PAR=16; PS=16; CTX=65536 ;; pc_*) PAR=$(echo $V | cut -d_ -f2); PS=$PAR; CTX=$(echo $V | cut -d_ -f3) ;; p16bf16) PAR=16; PS=16; KV=bf16; CTX=32768 ;; nodraft) DR="--no-drafts" ;; stock) MIA=0; PAR=4 ;; stock16) MIA=0; PAR=16; CTX=32768 ;; *) echo "variant?"; exit 2 ;; esac
  KVARG="--kv-cache $KV"; [ $MIA = 0 ] && KVARG=""
  ENVS="-e HF_HUB_OFFLINE=1 ${TFKERN:--e TENSORFOLD_CUDA_KERNELS=native} -e TENSORFOLD_NO_UPDATE_CHECK=1 -e TENSORFOLD_MEMORY_RESERVE_GIB=9"
  [ $MIA = 1 ] && ENVS="$ENVS -e TF_FLASHNEXT_HOST_DRAW_THREADS=7 -e TF_FLASHNEXT_HEAD_PIECE=1536 -e TF_FLASHNEXT_SAMPLED_HEADS=1 -e TF_FLASHNEXT_PROMPT_HC2=1 -e TF_FLASHNEXT_PROMPT_EXPERT_TILE_ROWS=256 -e TF_FLASHNEXT_KEEP_LATE=1 -e TF_FLASHNEXT_PRODUCT_STREAMS=$PS -e TF_FLASHNEXT_DRAFT_AHEAD=1 -e TF_FLASHNEXT_PREFILL_TAIL=512 -e TF_FLASHNEXT_PROMPT_SHARED_DIRECT=1 -e TF_FLASHNEXT_PROMPT_GEMM2=1 -e TF_FLASHNEXT_GATHER_EARLY=1 -e TF_FLASHNEXT_GDN_LANES=1 -e TF_FLASHNEXT_DRAFTS=${TFDRAFTS:-6} -e TF_FLASHNEXT_GATHER_READ=pread -e TF_FLASHNEXT_CONFIDENCE=${TFCONF:--0.4} -e TF_FLASHNEXT_SHARED_SIDE=1 -e TF_FLASHNEXT_WARM_AHEAD=1 -e TF_FLASHNEXT_SAMPLE_CAND=1"
  ENVS="$ENVS $TFENV"   # TFARGS="--vision ...": extra serve arguments. TFENV="-e NAME=value ...": extra engine environment (later -e wins over the defaults above)
  mkdir -p $Q/tf/kcache-$V $Q/prof; chmod 777 $Q/prof
  # TFVISION=1: image and video input as MiaAI's start.sh passes it (--vision and its three limits)
  [ -n "$TFVISION" ] && TFARGS="--vision --vision-max-images ${TFMAXIMG:-50} --vision-max-videos ${TFMAXVID:-4} --vision-image-tokens ${TFIMGTOK:-16384} $TFARGS"
  # TFWRAP="nsys launch -t cuda ...": run the server under a wrapper found in the image
  # TFENTRY=/opt/nvidia/nvidia_entrypoint.sh: start through the image's own entrypoint script (as MiaAI's start.sh does), then the engine
  if [ -n "$TFENTRY" ]; then ENTRY="--entrypoint $TFENTRY $IMG $EXE"; elif [ -n "$TFWRAP" ]; then ENTRY="--entrypoint ${TFWRAP%% *} $IMG ${TFWRAP#* } $EXE"; else ENTRY="--entrypoint $EXE $IMG"; fi
  docker run -d --name q38-run-tf --gpus all --ipc=host --network host --ulimit memlock=-1 --ulimit stack=67108864 $ENVS \
    -v $MODEL:/model:ro -v $Q/tf/kcache-$V:/cache -v $BINDIR:/opt/tensorfold:ro -v $Q/prof:/prof $ENTRY \
    serve /model --name Qwen3.8-Flash-Next --host 127.0.0.1 --port $PORT --context $CTX --parallel $PAR --max-tokens 32768 --temperature 1.0 --top-p 0.95 --top-k 20 --thinking $KVARG $DR $TFARGS >/dev/null
  ready /v1/models 900 q38-run-tf ;;
nvref)  # the model card's own stack: vLLM + b12x in a-vllm-image (command copied from nvidia/Qwen3.8-Flash-Next-FP4-Local README, host/port changed)
  M=/root/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/b9ce18a6887e18c089fdea8509a33e63429a3300
  SPEC='{"method":"mtp","num_speculative_tokens":4}'; [ $V = nodraft ] && SPEC=""
  mkdir -p $Q/nvref/vllm $Q/nvref/flashinfer $Q/nvref/triton $Q/nvref/b12x
  docker run -d --name q38-run-nvref --gpus all --ipc host --network host --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
    --security-opt seccomp=unconfined --security-opt apparmor=unconfined -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_PLE_TABLE_MEMORY=disk -e B12X_DISK_BACKEND=io_uring -e CUTE_DSL_ARCH=sm_121a -e SAFETENSORS_FAST_GPU=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_SSM_CONV_STATE_LAYOUT=DS -e VLLM_USE_AOT_COMPILE=1 -e VLLM_USE_MEGA_AOT_ARTIFACT=1 -e VLLM_USE_V2_MODEL_RUNNER=1 -e B12X_POLICY_MODE=auto \
    -e VLLM_MXFP8_LM_HEAD=1 -e VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13 -e VLLM_B12X_MOE_FP4_FORCE_A16=1 -e B12X_W4A16_A4_PREFILL=1 \
    -v $Q/hf:/root/.cache/huggingface:ro -v $Q/nvref/vllm:/root/.cache/vllm -v $Q/nvref/flashinfer:/root/.cache/flashinfer -v $Q/nvref/triton:/root/.triton -v $Q/nvref/b12x:/root/.cache/b12x \
    --entrypoint /bin/bash a-vllm-image:latest -c "vllm serve $M --served-model-name Qwen3.8-Flash-Next --host 127.0.0.1 --port $PORT --tensor-parallel-size 1 --pipeline-parallel-size 1 --mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill --dtype bfloat16 --kv-cache-dtype fp8 --quantization modelopt_mixed --block-size 16 --load-format b12x --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 4096 ${SPEC:+--speculative-config '$SPEC'} --gdn-decode-kernel b12x --linear-backend b12x --moe-backend b12x --no-enable-flashinfer-autotune --mm-encoder-tp-mode data --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice --compilation-config '{\"pass_config\":{\"fuse_act_quant\":true}}' --gpu-memory-utilization 0.8" >/dev/null
  ready /v1/models ${NVREF_WAIT:-1800} q38-run-nvref ;;
*) echo "engine?"; exit 2 ;;
esac
