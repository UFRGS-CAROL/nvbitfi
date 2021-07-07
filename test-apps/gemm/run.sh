#!/bin/bash

OPNUM=32
DMR=none
ALPHA=1.0
BETA=0.0
SIZE=512
DATADIR=/home/fernando/radiation-benchmarks/data/gemm

CUBLAS=0
PRECISION=float
CUDAPATH=/usr/local/cuda
if [ $# -gt 0 ]; then
  PRECISION=$1
  CUDAPATH=$2
  CUBLAS=$3
  if [ $CUBLAS -eq 1 ]; then
    USECUBLAS=--use_cublas
  fi
fi

eval LD_LIBRARY_PATH=${CUDAPATH}:$LD_LIBRARY_PATH ${PRELOAD_FLAG} ${BIN_DIR}/gemm ${USECUBLAS} --size ${SIZE} --precision ${PRECISION} --dmr ${DMR} --iterations 1 --alpha ${ALPHA} --beta ${BETA} --input_a ${DATADIR}/a_float_${ALPHA}_${BETA}_${SIZE}_cublas_${CUBLAS}_tensor_0.matrix --input_b ${DATADIR}/b_float_${ALPHA}_${BETA}_${SIZE}_cublas_${CUBLAS}_tensor_0.matrix --input_c ${DATADIR}/c_float_${ALPHA}_${BETA}_${SIZE}_cublas_${CUBLAS}_tensor_0.matrix --gold ${DATADIR}/g_float_${ALPHA}_${BETA}_${SIZE}_cublas_${CUBLAS}_tensor_0.matrix --opnum ${OPNUM} --verbose >stdout.txt 2>stderr.txt

sed -i '/LOGFILENAME/c\' stdout.txt
sed -i '/Time/c\' stdout.txt
sed -i '/time/c\' stdout.txt
sed -i '/^$/d' stdout.txt
