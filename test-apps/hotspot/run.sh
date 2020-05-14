#!/bin/bash

RADDIR=/home/carol/radiation-benchmarks/data/hotspot


eval ${PRELOAD_FLAG} ${BIN_DIR}/hotspot -verbose -size=1024 -sim_time=10 -streams=1 -input_temp=${RADDIR}/temp_1024 -input_power=${RADDIR}/power_1024 -gold_temp=${RADDIR}/gold_1024_1 -iterations=1 > stdout.txt 2> stderr.txt
sed -i '/LOGFILENAME/c\' stdout.txt 
sed -i '/Time/c\' stdout.txt 
sed -i '/time/c\' stdout.txt 
sed -i '/^$/d' stdout.txt
sed -i '/Performance/c\' stdout.txt
