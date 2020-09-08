#!/bin/bash

# Fault layer analysis
export FAULT_LAYER_PATH=/tmp/layer_path
mkdir -p ${FAULT_LAYER_PATH}


eval ${PRELOAD_FLAG} ${BIN_DIR}/darknet classifier predict ${BIN_DIR}/cfg/mnist.dataset ${BIN_DIR}/cfg/mnist_lenet.cfg ${BIN_DIR}/data/mnist_lenet.weights ${BIN_DIR}/data/mnist/images/v_00000_c7.png  > stdout.txt 2> stderr.txt
sed -i '/seconds/c\' stdout.txt 
sed -i '/seconds/c\' stderr.txt

# Current timestamp
timestamp=$(date +%s)

# Make sure destination exist
final_path=${BIN_DIR}/lenet_layers
mkdir -p ${final_path}

# Post layer movement
tar czf ${timestamp}.tar.gz ${FAULT_LAYER_PATH}/*.csv -C ${final_path}/
