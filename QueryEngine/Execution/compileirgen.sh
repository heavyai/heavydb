clear
g++ -g -std=c++11 -o testQPIRGen testQPIRGeneration.cpp QPIrGenerator.cpp QPIRPrepper.cpp ../Parse/RA/parser.cpp ../Parse/RA/scanner.cpp -I/usr/local/include -L/usr/local/lib -lpqxx `llvm-config --cppflags --ldflags --libs` -w 
