#!/bin/bash

OPNUM=1
DMR=none
ALPHA=1.0
BETA=0.0
SIZE=256
PRECISION=float
DATADIR=/home/carol/radiation-benchmarks/data/gemm


${BIN_DIR}/gemm --size ${SIZE} --precision ${PRECISION} --dmr ${DMR} --iterations 1 --alpha ${ALPHA} --beta ${BETA} --input_a ${DATADIR}/a_float_${ALPHA}_${BETA}_${SIZE}_cublas_0_tensor_0.matrix --input_b ${DATADIR}/b_float_${ALPHA}_${BETA}_${SIZE}_cublas_0_tensor_0.matrix --input_c ${DATADIR}/c_float_${ALPHA}_${BETA}_${SIZE}_cublas_0_tensor_0.matrix  --gold ${DATADIR}/g_float_${ALPHA}_${BETA}_${SIZE}_cublas_0_tensor_0.matrix --opnum ${OPNUM}  > stdout.txt 2> stderr.txt

sed -i '/LOGFILENAME/c\' stdout.txt 
sed -i '/Time/c\' stdout.txt 
sed -i '/time/c\' stdout.txt 
sed -i '/TESTE/c\' stdout.txt 
sed -i '/========/c\' stdout.txt
