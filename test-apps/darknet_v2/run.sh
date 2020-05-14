#!/bin/bash

RADDIR=/home/carol/radiation-benchmarks/data/darknet


eval ${PRELOAD_FLAG}  ${BIN_DIR}/darknet_v2 test_radiation -d ${RAD_DIR}/data/darknet/fault_injection.csv -n 1 -s 0 -a 0 > stdout.txt 2> stderr.txt
sed -i '/Iteration 0 - image/c\' stdout.txt 
sed -i '/^$/d' stdout.txt

