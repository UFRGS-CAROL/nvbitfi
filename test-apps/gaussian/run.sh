#!/bin/bash

RADDIR=/home/carol/radiation-benchmarks
DIR=${RADDIR}/data/gaussian

SIZE=512

eval ${PRELOAD_FLAG} ${APP_DIR}/cudaGaussian --size ${SIZE} --input ${RADDIR}/input_${SIZE}.data --gold ${RADDIR}/gold_${SIZE}.data --iterations 1 --verbose > stdout.txt 2>stderr.txt
sed -i '/time/c\' stdout.txt 
sed -i '/LOGFILENAME/c\' stdout.txt
sed -i '/^$/d' stdout.txt
