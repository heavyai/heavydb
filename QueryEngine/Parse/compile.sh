bison++ -d -hparser.h -o parser.cpp parser.y
flex++ -d -i -oscanner.cpp scanner.l
g++ -O3 -c parser.cpp -w
g++ -O3 -c scanner.cpp -w
g++ -O3 -c visitorTest.cpp -w
g++ -O3 -c perfTest.cpp -w
g++ -O3 -o visitorTest visitorTest.o parser.o scanner.o -w 
g++ -O3 -o perfTest perfTest.o parser.o scanner.o -w -lboost_timer -lboost_system
