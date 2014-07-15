cd SQL
bison++ -d -hparser.h -o parser.cpp parser.y
flex++ -d -i -oscanner.cpp scanner.l
cd ../RA
bison++ -d -hRelAlgebraParser.h -o RelAlgebraParser.cpp RelAlgebraParser.y
flex++ -d -i -oRelAlgebraScanner.cpp RelAlgebraLexer.l
cd ..
g++ visitorTest.cpp