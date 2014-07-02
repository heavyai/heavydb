
testShort: parserShort.o scannerShort.o test.o 
	g++ -o testShort test.o parserShort.o scannerShort.o

parserShort.o: parserShort.cpp
	g++ -c parserShort.cpp

test.o: test.cpp
	g++ -c test.cpp

scannerShort.o:	scannerShort.cpp
	g++ -c scannerShort.cpp

parserShort.cpp: parserShort.y
	bison++ -d -hparserShort.h -o parserShort.cpp parserShort.y

scannerShort.cpp: scannerShort.l
	flex++ -d -oscannerShort.cpp scannerShort.l