RM = rm -f
OPTFLAGS = -O2 -Wall -pedantic
LDFLAGS =  

CXX = mpicc $(OPTFLAGS) -g -fopenmp
CC = gcc $(OPTFLAGS) -g

EXEC = mandel 00openmp_mandel 01openmp_master_worker  01openmp_work_pool
BINS = *.o *~  *.out


.PHONY: clean fclean dclean iclean

all: $(EXEC)

mandel: mandel.c
	$(CC) -o mandel  mandel.c

00openmp_mandel: 00openmp_mandel.c
	$(CXX) -o 00openmp_mandel 00openmp_mandel.c

01openmp_master_worker: 01openmp_master_worker.c
	$(CXX) -o 01openmp_master_worker 01openmp_master_worker.c

01openmp_work_pool: 01openmp_work_pool.c
	$(CXX) -o 01openmp_work_pool 01openmp_work_pool.c

	
clean:
	$(RM) $(BINS)

fclean: clean
	$(RM) $(EXEC)

dclean:
	$(RM) Data/*.dat

iclean:
	$(RM) Image/*.ppm