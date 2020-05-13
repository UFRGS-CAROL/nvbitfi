#!/bin/bash

RADDIR=/home/carol/radiation-benchmarks/data/quicksort

SIZE=1048576 

eval ${PRELOAD_FLAG} ${BIN_DIR}/quicksort -size=${SIZE} -input=${RADDIR}/quicksort_input_134217728 -gold=${RADDIR}/quicksort_gold_${SIZE} -iterations=1 -verbose > stdout.txt 2> stderr.txt
sed -i '/LOGFILENAME/c\' stdout.txt 
sed -i '/Time/c\' stdout.txt 
sed -i '/time/c\' stdout.txt 
sed -i '/^$/d' stdout.txt
sed -i '/Perf/c\' stdout.txt
sed -i '/Starting/c\' stdout.txt
sed -i '/Done in/c\' stdout.txt
