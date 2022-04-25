CC = nvcc
PROGRAM = program.out
MAIN = main.cu
INCS = include/helper.cuh include/conv_cpu.cuh include/conv_gpu_naive.cuh include/conv_gpu_im2col.cuh 
INC_PATH = ./include/
COMPILE_OPTION = 
DEBUG=OFF

.PHONY : all run clean

all: ${PROGRAM}

${PROGRAM}: ${MAIN} ${INCS} Makefile
	${CC} -o $@ ${MAIN} ${COMPILE_OPTION} -I${INC_PATH} -DDEBUG_${DEBUG}

run : ${PROGRAM}
	./${PROGRAM}

clean :
	rm ${PROGRAM}