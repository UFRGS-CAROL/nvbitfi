#!/bin/bash


eval ${PRELOAD_FLAG} ${BIN_DIR}/darknet classifier predict ${BIN_DIR}/cfg/mnist.dataset ${BIN_DIR}/cfg/mnist_lenet.cfg ${BIN_DIR}/data/mnist_lenet.weights ${BIN_DIR}/data/mnist/images/v_00000_c7.png  > stdout.txt 2> stderr.txt
sed -i '/seconds/c\' stdout.txt 
sed -i '/seconds/c\' stderr.txt
