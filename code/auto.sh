#!/bin/bash
COMMAND=$1
for i in {1..30}
do
	OMP_DYNAMIC=false $COMMAND < input
done
