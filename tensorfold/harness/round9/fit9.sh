#!/bin/bash
# fit9.sh (923b, engine stopped): evalrot.py (old vs new rotation on fit / fresh / held rows), then builder B's mkhead2.py on the fitting rows
# -> head2_r9.bin + .rot, then mkhead2.py from that .rot alone (what `recipe.sh head2` does) and a byte comparison of the two files.
Q=$HOME/q38; K9=$Q/kit-A9; H=$Q/repo_tf/head2; mkdir -p $Q/head2r9
S=$(basename $(ls -d $Q/hf/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/* | head -1)); M=/hf/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/$S
run() { docker run --rm --name q38fit-A --gpus all --ipc=host -v $Q/hf:/hf:ro -v $H:/h:ro -v $K9:/k:ro -v $Q/prof:/prof -v $Q/runs/R9_dump:/rd:ro -v $Q/head2r9:/out --entrypoint python3 nvcr.io/nvidia/pytorch:26.07-py3 "$@" 2>&1 | grep -v "^$\|NVIDIA\|Copyright\|license\|WARNING\|pulled\|container image" ; }
run /k/evalrot.py $M /h/draft_vocab_default.txt /prof/r9_all.mix /rd/engine.log 260 123 /h/head2_rot_r512.rot /prof/r9 /k/corpus/fresh.jsonl
run /h/mkhead2.py $M /h/draft_vocab_default.txt /prof/r9_fit.mix /out/head2_r9.bin 512
cp $Q/head2r9/head2_r9.bin.rot $Q/head2r9/head2_rot_r9.rot
run /h/mkhead2.py $M /h/draft_vocab_default.txt /out/head2_rot_r9.rot /out/head2_r9_fromrot.bin 512
sha256sum $Q/head2r9/* $H/head2_rot_r512.rot $Q/hf/q38/head2_r512.bin | cut -c1-100; ls -la $Q/head2r9
cmp $Q/head2r9/head2_r9.bin $Q/head2r9/head2_r9_fromrot.bin && echo "FROM_ROT_IDENTICAL"
