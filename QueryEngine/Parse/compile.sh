bison++ -d -hparser.h -o parser.cpp parser.y
flex++ -d -oscanner.cpp scanner.l
g++ -c parser.cpp -w
g++ -c scanner.cpp -w
g++ -c visitorTest.cpp -w
g++ -o visitorTest visitorTest.o parser.o scanner.o -w
