#!/bin/bash
COMMAND=$1
THREADS=$2
for i in {1..30}
do
	OMP_DYNAMIC=false ./$COMMAND.out < input$THREADS
done
