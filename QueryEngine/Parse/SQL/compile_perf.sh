bison++ -d -hparser.h -o parser.cpp parser.y
flex++ -d -i -oscanner.cpp scanner.l
g++ -O3 parser.cpp -c -w
g++ -O3 scanner.cpp -c -w
g++ -O3 -c perfTest.cpp -w -I/usr/local/include
g++ -O3 -o perfTest perfTest.o parser.o scanner.o -w -L/usr/local/lib -lboost_timer -lboost_system 
rm perfTest.o scanner.o parser.o
