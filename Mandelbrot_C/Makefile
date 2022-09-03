RM = rm -f
OPTFLAGS = -O2 -Wall -pedantic
CFLAGS = $(OPTFLAGS) -g
LDFLAGS =

CC = gcc $(CFLAGS)
EXEC = mandel mandel_load_balance
BINS = *.o *~ *.out


.PHONY: clean fclean dclean iclean

all: $(EXEC)


mandel: mandel.c
	$(CC) -o mandel  mandel.c

mandel_load_balance: mandel_load_balance.c
	$(CC) -o mandel_load_balance  mandel_load_balance.c


clean:
	$(RM) $(BINS)

fclean: clean
	$(RM) $(EXEC)

dclean: 
	$(RM) Data/*.dat

iclean: 
	$(RM) Image/*.ppm
