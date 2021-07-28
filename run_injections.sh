#!/bin/bash

# stop after first error
set -e

if [ $# -ne 4 ]; then
  echo "$0 <BENCHMARK> <FAULTS> <ADDITIONAL PARAMETERS LIST between \"\">"
  exit 1
fi

export BENCHMARK=$1
export FAULTS=$2
export ADDITIONAL_PARAMETERS=$3

./test.sh "$1"
exit 0