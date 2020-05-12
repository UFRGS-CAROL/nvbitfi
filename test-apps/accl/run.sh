#!/bin/bash

RADD=/home/carol/radiation-benchmarks/data/accl

#${BIN_DIR}/accl 2 1  ${APP_DIR}/2Frames.pgm ${APP_DIR}/GOLD_2Frames 1 -verbose > stdout.txt 2> stderr.txt 
eval ${PRELOAD_FLAG}  ${BIN_DIR}/cudaACCL --size 7 --frames 7 --input ${RADD}/7Frames.pgm --gold ${RADD}/gold_7_7.data --iterations 1 --verbose  > stdout.txt 2>stderr.txt
sed -i '/LOGFILENAME/c\' stdout.txt 
sed -i '/Time/c\' stdout.txt 
sed -i '/time/c\' stdout.txt 
sed -i '/^$/d' stdout.txt

