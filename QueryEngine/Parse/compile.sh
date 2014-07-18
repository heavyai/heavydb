cd SQL
bison++ -d -hparser.h -o parser.cpp parser.y
flex++ -d -i -oscanner.cpp scanner.l
g++ -O3 parser.cpp -c -w
g++ -O3 scanner.cpp -c -w
cd ..
g++ -O3 -c perfTest.cpp -w
g++ -O3 -o perfTest perfTest.o SQL/parser.o SQL/scanner.o -w -lboost_timer -lboost_system
rm perfTest.o SQL/scanner.o SQL/parser.o
