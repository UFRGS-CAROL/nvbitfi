#!/bin/bash

# stop after first error
set -e

#lava_mp gemm_tensorcores bfs accl mergesort quicksort hotspot darknet_v2 gaussian lud nw
for i in gemm_tensorcores; 
do
    echo "###############################################################"
    echo "                     DOING FOR $i"
    echo "###############################################################"
    ./test.sh ${i}
 done;
