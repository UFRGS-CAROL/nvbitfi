#!/bin/bash

# stop after first error
set -e

# it still missing darknet_v2 darknet_v3 py_faster_rcnn resnet_torch gemm_tensorcores
benchmarks=( lava_mp gemm_tensorcores bfs accl mergesort quicksort hotspot gaussian lud nw )

for i in "${benchmarks[@]}"
do
    echo $i
    make -C test-apps/${i}/ -j4
    make -C test-apps/${i}/ generate
    make -C test-apps/${i}/ test
done


for i in "${benchmarks[@]}"; 
do
    echo "###############################################################"
    echo "                     DOING FOR $i"
    echo "###############################################################"
    export BENCHMARK=${i}
    export FAULTS=2 #000
    ./test.sh ${i}

    tar czf ${i}_nvbitfi_1k.tar.gz logs_sdcs_* /home/carol/NVBITFI/nvbit_release/tools/nvbitfi/logs/results/ /var/radiation-benchmarks/log/
    rm -rf /var/radiation-benchmarks/log/*.log logs/* *.csv
done;

