

CUDA_PATH = /usr/local/cuda/
CUDA_DEBUG = -g -G -lineinfo

NVCC_OPT = --ptxas-options=-v

NVCC = nvcc
CC = gcc -O2 -Wall -pedantic -g

EXEC = mandel 01cuda_mandel 02cuda_mandel test

BINS = *.o *~ *.out


.PHONY: clean fclean dclean iclean

all: $(EXEC)

mandel: mandel.c
	$(CC) -o mandel mandel.c

01cuda_mandel: 01cuda_mandel.cu
	$(NVCC) -o 01cuda_mandel 01cuda_mandel.cu

02cuda_mandel: 02cuda_mandel.cu
	$(NVCC) -o 02cuda_mandel 02cuda_mandel.cu


clean:
	$(RM) $(BINS)

fclean: clean
	$(RM) $(EXEC)

dclean:
	$(RM) Data/*.dat

iclean:
	$(RM) Image/*.ppm