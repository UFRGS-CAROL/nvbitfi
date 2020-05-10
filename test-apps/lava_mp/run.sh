#!/bin/bash

OPNUM=194
DMR=none
SIZE=2 #3
PRECISION=single
DATADIR=/home/carol/radiation-benchmarks/data/lava
#echo ${PRELOAD_FLAG}
#PRELOAD_FLAG="LD_PRELOAD=/home/carol/NVBITFI/nvbit_release/tools/nvbitfi/injector/injector.so"
#BIN_DIR=./
eval ${PRELOAD_FLAG} ${BIN_DIR}/cuda_lava -boxes ${SIZE} -streams 1 -iterations 1 -input_distances ${DATADIR}/lava_${PRECISION}_distances_${SIZE} -input_charges ${DATADIR}/lava_${PRECISION}_charges_${SIZE} -output_gold ${DATADIR}/lava_${PRECISION}_gold_${SIZE} -opnum ${OPNUM} -precision ${PRECISION} -redundancy ${DMR} > stdout.txt 2> stderr.txt
