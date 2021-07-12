#!/bin/bash

SIZE=1024
SIM_TIME=100
STREAMS=1
PRECISION=float
CUDAPATH=/usr/local/cuda

if [ $# -gt 0 ]; then
  CUDAPATH=$1
  PRECISION=$2
  STREAMS=$3
  SIM_TIME=$4
fi

RADDIR=/home/carol/radiation-benchmarks/data/hotspot

eval LD_LIBRARY_PATH=${CUDAPATH}/lib64:$LD_LIBRARY_PATH ${PRELOAD_FLAG} ${BIN_DIR}/cuda_hotspot -verbose -size=${SIZE} -sim_time=${SIM_TIME} -streams=${STREAMS} -input_temp=${RADDIR}/temp_${SIZE} -input_power=${RADDIR}/power_${SIZE} -gold_temp=${RADDIR}/gold_${SIZE}_${SIM_TIME} -iterations=1 > stdout.txt 2> stderr.txt
sed -i '/LOGFILENAME/c\' stdout.txt 
sed -i '/Time/c\' stdout.txt 
sed -i '/time/c\' stdout.txt 
sed -i '/^$/d' stdout.txt
sed -i '/Performance/c\' stdout.txt
