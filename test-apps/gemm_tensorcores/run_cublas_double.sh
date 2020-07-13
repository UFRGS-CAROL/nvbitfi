#!/bin/bash

OPNUM=1
DMR=none
ALPHA=1.0
BETA=0.0
SIZE=4096
PRECISION=double
DATADIR=/home/carol/radiation-benchmarks/data/gemm

eval ${PRELOAD_FLAG} ${BIN_DIR}/gemm --size ${SIZE} --precision ${PRECISION} --dmr ${DMR} --iterations 1 --alpha ${ALPHA} --beta ${BETA} --input_a ${DATADIR}/a_${PRECISION}_${ALPHA}_${BETA}_${SIZE}_cublas_1_tensor_0.matrix --input_b ${DATADIR}/b_${PRECISION}_${ALPHA}_${BETA}_${SIZE}_cublas_1_tensor_0.matrix --input_c ${DATADIR}/c_${PRECISION}_${ALPHA}_${BETA}_${SIZE}_cublas_1_tensor_0.matrix  --gold ${DATADIR}/g_${PRECISION}_${ALPHA}_${BETA}_${SIZE}_cublas_1_tensor_0.matrix --opnum ${OPNUM} --use_cublas --verbose > stdout.txt 2> stderr.txt

sed -i '/LOGFILENAME/c\' stdout.txt 
sed -i '/Time/c\' stdout.txt 
sed -i '/time/c\' stdout.txt 
sed -i '/^$/d' stdout.txt 
