#!/bin/bash

SIZE=256
DATADIR=/home/carol/radiation-benchmarks/data/gemm
PRECISION=half

eval ${PRELOAD_FLAG} ${BIN_DIR}/cuda_trip_mxm_${PRECISION} -size=${SIZE} -input_a=${DATADIR}/mxm_${PRECISION}_A_8192.matrix -input_b=${DATADIR}/mxm_${PRECISION}_B_8192.matrix -gold=${DATADIR}/mxm_${PRECISION}_GOLD_.matrix  -iterations=1 -verbose -no-warmup > stdout.txt 2> stderr.txt
sed -i '/Preparing/c\' stdout.txt
sed -i '/Average/c\' stdout.txt
sed -i '/time/c\' stdout.txt
sed -i '/SIZE/c\' stdout.txt

