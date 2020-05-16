#!/bin/bash

RADDIR=/home/carol/radiation-benchmarks
DATA=${RADDIR}/data/nw

#./nw 16384 10 ./input_16384 ./gold_16384 1 0
SIZE=16384

eval ${PRELOAD_FLAG}  ${BIN_DIR}/nw ${SIZE} 10 ${DATA}/input_${SIZE} ${DATA}/gold_${SIZE} 1 0 > stdout.txt 2> stderr.txt
sed -i '/kernel time/c\REPLACED.' stdout.txt 
sed -i '/LOGFILE/c\REPLACED.' stdout.txt 
sed -i '/read../c\REPLACED.' stdout.txt 
sed -i '/^$/d' stdout.txt
