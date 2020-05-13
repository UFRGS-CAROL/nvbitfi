#!/bin/bash

RADDIR=/home/carol/radiation-benchmarks/data/mergesort


eval ${PRELOAD_FLAG}  ${BIN_DIR}/mergesort -size=1048576 -input=${RADDIR}/mergesort_input_134217728 -gold=${RADDIR}/mergesort_gold_1048576 -iterations=1 -verbose > stdout.txt 2> stderr.txt
sed -i '/LOGFILENAME/c\' stdout.txt 
sed -i '/Time/c\' stdout.txt 
sed -i '/time/c\' stdout.txt 
sed -i '/^$/d' stdout.txt
sed -i '/Perf/c\' stdout.txt
sed -i '/Starting/c\' stdout.txt
