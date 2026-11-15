#!/bin/bash
# tfbuild.sh NAME [CPUS] [MEM] : build tensorfold-native + fatbins from ~/q38/build/tf into ~/q38/build/out/NAME (a full TFBIN dir).
# Runs in NVIDIA's PyTorch container (nvcc), Zig 0.17.0 pinned by sha256, capped CPU and memory. No GPU needed.
N=$1; CPUS=${2:-8}; MEM=${3:-7g}; B=$HOME/q38/build; mkdir -p $B/out $B/zcache
docker rm -f q38-build >/dev/null 2>&1
docker run --rm --name q38-build --cpus $CPUS --memory $MEM --memory-swap $MEM --network host -v $B:/b -e ZIG_GLOBAL_CACHE_DIR=/b/zcache \
  --entrypoint bash nvcr.io/nvidia/pytorch:26.07-py3 -c '
set -e
if [ ! -x /b/zig/zig ]; then
  curl -fsSL -o /tmp/zig.tar.xz https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz
  echo "9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  /tmp/zig.tar.xz" | sha256sum -c -
  mkdir -p /b/zig && tar -xJf /tmp/zig.tar.xz -C /b/zig --strip-components=1
fi
cd /b/tf && rm -rf /b/out/'$N' && /b/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=fast --prefix /b/out/'$N' -j'$CPUS' fatbins native 2> /b/out/'$N'.err || { echo BUILD_FAILED; grep -E "error|Killed|Maximum resident" /b/out/'$N'.err | head -40; exit 1; }
true
chmod -R a+rX /b/out/'$N'; chown -R '$(id -u):$(id -g)' /b/out/'$N' /b/out/'$N'.err' 2>&1 | tail -45
O=$B/out/$N
if [ -x $O/native/bin/tensorfold-native ]; then
  V=$HOME/q38/tfbin/v114; mkdir -p $O/native/lib $O/lib $O/share
  cp -an $V/lib/. $O/lib/ 2>/dev/null; cp -an $V/native/lib/. $O/native/lib/ 2>/dev/null; cp -an $V/share/. $O/share/ 2>/dev/null
  echo "BUILD_OK $N $(sha256sum $O/native/bin/tensorfold-native | cut -c1-16) $(ls $O/fatbin 2>/dev/null | wc -l) fatbins"
fi
