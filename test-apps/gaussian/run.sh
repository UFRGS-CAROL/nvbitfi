#!/bin/bash

RADDIR=/home/carol/radiation-benchmarks
DIR=${RADDIR}/data/gaussian
CUDAPATH=/usr/local/cuda
SIZE=512
if [ $# -gt 0 ]; then
  CUDAPATH=$1
  SIZE=$2
fi


eval LD_LIBRARY_PATH=${CUDAPATH}/lib64:$LD_LIBRARY_PATH  ${PRELOAD_FLAG} ${APP_DIR}/cudaGaussian --size ${SIZE} --input ${RADDIR}/input_${SIZE}.data --gold ${RADDIR}/gold_${SIZE}.data --iterations 1 --verbose > stdout.txt 2>stderr.txt
sed -i '/time/c\' stdout.txt 
sed -i '/LOGFILENAME/c\' stdout.txt
sed -i '/^$/d' stdout.txt
