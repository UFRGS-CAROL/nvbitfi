#!/bin/bash

OPNUM=194
DMR=none
SIZE=2 #3
PRECISION=single
DATADIR=/home/carol/radiation-benchmarks/data/lava

eval ${PRELOAD_FLAG} ${BIN_DIR}/cuda_lava -boxes ${SIZE} -streams 1 -iterations 1 -input_distances ${DATADIR}/lava_${PRECISION}_distances_${SIZE} -input_charges ${DATADIR}/lava_${PRECISION}_charges_${SIZE} -output_gold ${DATADIR}/lava_${PRECISION}_gold_${SIZE} -opnum ${OPNUM} -precision ${PRECISION} -redundancy ${DMR} -verbose > stdout.txt 2> stderr.txt
sed -i '/LOGFILENAME/c\' stdout.txt 
sed -i '/time/c\' stdout.txt 
sed -i '/TESTE/c\' stdout.txt 
sed -i '/========/c\' stdout.txt

