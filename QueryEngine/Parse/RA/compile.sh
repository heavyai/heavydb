bison++ -d -hparser.h -o parser.cpp parser.y
flex++ -d -i -oscanner.cpp scanner.l
g++ scanner.cpp parser.cpp xml.cpp -o xml -w
g++ scanner.cpp parser.cpp queryPlan.cpp -o queryPlan -w

