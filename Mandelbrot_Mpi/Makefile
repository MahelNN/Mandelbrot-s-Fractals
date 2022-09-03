RM = rm -f
OPTFLAGS = -O2 -Wall -pedantic
CFLAGS = $(OPTFLAGS) -g
CXXFLAGS = $(OPTFLAGS) -g
LDFLAGS =

CXX = mpicc $(CXXFLAGS)
CC = gcc $(CFLAGS)

EXEC = 01mpi_block 02mpi_cyclic_wbcomm 03mpi_cyclic_wasync 04mpi_cyclic_wpacking mandel
BINS = *.o *~  *.out


.PHONY: clean fclean dclean iclean

all: $(EXEC)

mandel: mandel.c
	$(CC) -o mandel  mandel.c

01mpi_block: 01mpi_block.c
	$(CXX) -o 01mpi_block  01mpi_block.c

02mpi_cyclic_wbcomm: 02mpi_cyclic_wbcomm.c
	$(CXX) -o 02mpi_cyclic_wbcomm  02mpi_cyclic_wbcomm.c

03mpi_cyclic_wasync: 03mpi_cyclic_wasync.c
	$(CXX) -o 03mpi_cyclic_wasync  03mpi_cyclic_wasync.c

04mpi_cyclic_wpacking: 04mpi_cyclic_wpacking.c
	$(CXX) -o 04mpi_cyclic_wpacking  04mpi_cyclic_wpacking.c


clean:
	$(RM) $(BINS)

fclean: clean
	$(RM) $(EXEC)

dclean: 
	$(RM) Data/*.dat

iclean:
	$(RM) Image/*.ppm