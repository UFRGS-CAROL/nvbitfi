#!/bin/bash

RADDIR=/home/carol/radiation-benchmarks/data/bfs

eval ${PRELOAD_FLAG}  ${BIN_DIR}/cudaBFS --input ${RADDIR}/graph1MW_6.txt --gold ${RADDIR}/gold.data --iterations 1 --verbose  > stdout.txt 2> stderr.txt
sed -i '/LOGFILENAME/c\' stdout.txt 
sed -i '/Time/c\' stdout.txt 
sed -i '/time/c\' stdout.txt 
sed -i '/^$/d' stdout.txt 
