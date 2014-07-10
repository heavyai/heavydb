bison++ -d -hRelAlgebraParser.h -o RelAlgebraParser.cpp RelAlgebraParser.y
flex++ -d -i -oRelAlgebraScanner.cpp RelAlgebraLexer.l
g++ RelAlgebraScanner.cpp RelAlgebraParser.cpp