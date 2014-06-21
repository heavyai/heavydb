CC=g++
CFLAGS=-Wall -c
LFLAGS=-Wall
DEBUG=-g
OBJS=File.o
SHARED=../../Shared

all:	    $(OBJS)

test:	    $(OBJS) fileTest.cpp
		$(CC) $(LFLAGS) $(OBJS) fileTest.cpp -o test

File.o:	    File.h File.cpp other
	    $(CC) $(CFLAGS) File.cpp

other:	    $(SHARED)/errors.h $(SHARED)/types.h

clean:
	\rm *.o *~
