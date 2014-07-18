g++ -o SQLParseTest SQLParseTest.cpp SQLParse.cpp SQL/parser.cpp SQL/scanner.cpp -w
g++ -o RAParseTest RAParseTest.cpp RAParse.cpp RA/parser.cpp RA/scanner.cpp -w
./SQLParseTest
./RAParseTest
