#!/bin/bash

#./nw 16384 10 ./input_16384 ./gold_16384 1 0
SIZE=16384
PENALTY=10
if [ $# -gt 0 ]; then
  CUDAPATH=$1
  SIZE=$2
  PENALTY=$3
fi

RADDIR=/home/carol/radiation-benchmarks
DATA=${RADDIR}/data/nw

eval LD_LIBRARY_PATH=${CUDAPATH}/lib64:$LD_LIBRARY_PATH ${PRELOAD_FLAG} ${BIN_DIR}/nw ${SIZE} ${PENALTY} ${DATA}/input_${SIZE}_${PENALTY} ${DATA}/gold_${SIZE}_${PENALTY} 1 0 > stdout.txt 2> stderr.txt
sed -i '/kernel time/c\REPLACED.' stdout.txt 
sed -i '/LOGFILE/c\REPLACED.' stdout.txt 
sed -i '/read../c\REPLACED.' stdout.txt 
sed -i '/^$/d' stdout.txt
