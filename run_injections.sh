#!/bin/bash

# stop after first error
set -e

export BENCHMARK=$1
export FAULTS=$2
./test.sh "$1"
