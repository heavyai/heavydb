bison++ -d -hparser.h -o parser.cpp parser.y
flex++ -d -i -oscanner.cpp scanner.l
g++ scanner.cpp parser.cpp tests/visitorTest.cpp -I/usr/local/include -L/usr/local/lib/ -lpqxx -o visitorTest -w -g -std=c++11

