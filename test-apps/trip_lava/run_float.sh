#!/bin/bash

SIZE=2
DATADIR=/home/carol/radiation-benchmarks/data/lava
PRECISION=single

eval ${PRELOAD_FLAG} ${BIN_DIR}/cuda_trip_lava_${PRECISION} -boxes=${SIZE} -streams=1 -iterations=1 -verbose -input_distances=${DATADIR}/lava_${PRECISION}_distances_${SIZE} -input_charges=${DATADIR}/lava_${PRECISION}_charges_${SIZE} -output_gold=${DATADIR}/lava_${PRECISION}_gold_${SIZE}  > stdout.txt 2> stderr.txt
sed -i '/Iteration/c\' stdout.txt

