#!/bin/bash
#set -x

if [ $# -gt 0 ]; then
  CUDAPATH=$1
  SIZE=$2
  FRAMES=$3
fi

RADD=/home/carol/radiation-benchmarks/data/accl

eval LD_LIBRARY_PATH=${CUDAPATH}/lib64:$LD_LIBRARY_PATH ${PRELOAD_FLAG} ${BIN_DIR}/cudaACCL --size ${SIZE} --frames ${FRAMES} --input ${RADD}/${FRAMES}Frames.pgm --gold ${RADD}/gold_${FRAMES}_${FRAMES}.data --iterations 1 --verbose > stdout.txt 2>stderr.txt
sed -i '/LOGFILENAME/c\' stdout.txt 
sed -i '/Time/c\' stdout.txt 
sed -i '/time/c\' stdout.txt 
sed -i '/^$/d' stdout.txt

