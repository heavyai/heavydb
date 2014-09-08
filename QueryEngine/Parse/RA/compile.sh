bison++ -d -hparser.h -o parser.cpp parser.y
flex++ -d -i -oscanner.cpp scanner.l
g++ scanner.cpp parser.cpp xml.cpp -o xml -w -g -std=c++0x -I/usr/local/include
g++ visitor/QPTranslator.cpp scanner.cpp parser.cpp queryPlan.cpp -o queryPlan -w -g -I/usr/local/include -std=c++0x

