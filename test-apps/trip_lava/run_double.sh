#!/bin/bash

SIZE=2
DATADIR=/home/carol/radiation-benchmarks/data/lava
PRECISION=double

eval ${PRELOAD_FLAG} ${BIN_DIR}/cuda_trip_lava_single -boxes=${SIZE} -streams=1 -iterations=1 -verbose -input_distances=${DATADIR}/lava_single_distances_${SIZE} -input_charges=${DATADIR}/lava_single_charges_${SIZE} -output_gold=${DATADIR}/lava_single_gold_${SIZE}  > stdout.txt 2> stderr.txt
sed -i '/Iteration/c\' stdout.txt

