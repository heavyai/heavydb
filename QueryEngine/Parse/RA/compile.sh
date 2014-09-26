bison++ -d -hparser.h -o parser.cpp parser.y
flex++ -d -i -oscanner.cpp scanner.l
g++ scanner.cpp parser.cpp tests/ra2xml.cpp visitor/XMLTranslator.cpp -I/usr/local/include -L/usr/local/lib/ -lpqxx -o ra2xml -w -g -std=c++11
