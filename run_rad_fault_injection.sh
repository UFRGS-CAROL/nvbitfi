#!/bin/bash

# stop after first error
set -e

#lava_mp gemm_tensorcores bfs accl mergesort quicksort hotspot darknet_v2 darknet_v3 gaussian lud nw
for i in gaussian lud nw; 
do
    echo "###############################################################"
    echo "                     DOING FOR $i"
    echo "###############################################################"
    export BENCHMARK=${i}
    export FAULTS=1000
    ./test.sh ${i}

    tar czf ${i}_nvbitfi_1k.tar.gz logs_sdcs_* /home/carol/NVBITFI/nvbit_release/tools/nvbitfi/logs/results/ /var/radiation-benchmarks/log/
    rm -rf /var/radiation-benchmarks/log/*.log logs/* *.csv
done;

sudo shutdown -h now
