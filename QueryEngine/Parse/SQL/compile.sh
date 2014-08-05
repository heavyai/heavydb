clear
bison++ -d -hparser.h -o parser.cpp parser.y
flex++ -d -i -oscanner.cpp scanner.l
g++ scanner.cpp parser.cpp visitorTest.cpp -o SQLParser -w -std=c++11
