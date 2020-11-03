PROG=jacobi

GCC ?= gcc
FLAGS_OPENMP ?= -fopenmp

debug:
	$(GCC) -Wall -g $(FLAGS_OPENMP) -O0 ${PROG}.c -o ${PROG}.out -lm

release:
	$(GCC) -Wall -g $(FLAGS_OPENMP) -O3 ${PROG}.c -o ${PROG}.out -lm

run go:
	OMP_DYNAMIC=false ./${PROG}.out < input

clean:
	rm -f ${PROG}.out ${PROG}.o *~
