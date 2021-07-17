#!/bin/bash

RADDIR=/home/carol/radiation-benchmarks
DATA=${RADDIR}/data/lud

#SIZE=8192
SIZE=2048
if [ $# -gt 0 ]; then
  CUDAPATH=$1
  SIZE=$2
fi

eval LD_LIBRARY_PATH=${CUDAPATH}/lib64:$LD_LIBRARY_PATH ${PRELOAD_FLAG}  ${BIN_DIR}/cudaLUD --size ${SIZE} --input ${DATA}/input_${SIZE}.data --gold ${DATA}/gold_${SIZE}.data --iterations 1 --verbose > stdout.txt 2> stderr.txt
sed -i '/Array set time/c\REPLACED.' stdout.txt 
sed -i '/Gold check time/c\REPLACED.' stdout.txt 
sed -i '/OUTPUT/c\REPLACED.' stdout.txt 
sed -i '/Iteration 0 overall time/c\REPLACED.' stdout.txt 
sed -i '/Total kernel time/c\REPLACED.' stdout.txt 
sed -i '/Average kernel time/c\REPLACED.' stdout.txt 
sed -i '/Device kernel time/c\REPLACED.' stdout.txt 
sed -i '/Copy time/c\REPLACED.' stdout.txt 

