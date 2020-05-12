#!/bin/bash

# stop after first error
set -e

#lava_mp gemm_tensorcores bfs accl mergesort quicksort hotspot darknet_v2 gaussian lud nw
for i in bfs; 
do
    echo "###############################################################"
    echo "                     DOING FOR $i"
    echo "###############################################################"
    export BENCHMARK=${i}
    export FAULTS=10 #00
    ./test.sh ${i}
done;
