#!/bin/bash

DATADIR=/home/carol/radiation-benchmarks/data/darknet
NETDIR=/home/carol/radiation-benchmarks/data
PRECISION=double

eval ${PRELOAD_FLAG} ${BIN_DIR}/darknet_v3_${PRECISION} detector test_radiation cfg/coco.data ${DATADIR}/yolov3-spp.cfg ${DATADIR}/yolov3-spp.weights \
                ${NETDIR}/networks_img_list/fault_injection.txt \
                -generate 0 -gold ${DATADIR}/test_${PRECISION}.csv -iterations 1 -norm_coord 0 -tensor_cores 0 -smx_redundancy 1 >stdout.txt 2>stderr.txt

sed -i '/Iteration/c\' stdout.txt

