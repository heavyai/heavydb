CC=g++
CFLAGS=-Wall -c
LFLAGS=-Wall
DEBUG=-g
OBJS=File.o
SHARED=../../Shared
TESTNAME=fileTest

all:	    $(OBJS)

test:	    $(OBJS) $(TESTNAME).cpp
		$(CC) $(LFLAGS) $(OBJS) $(TESTNAME).cpp -o $(TESTNAME)

File.o:	    File.h File.cpp other
	    $(CC) $(CFLAGS) File.cpp

other:	    $(SHARED)/errors.h $(SHARED)/types.h

clean:
	\rm *.o *~
