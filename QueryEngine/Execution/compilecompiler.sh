clear
g++ -g -std=c++11 -o testQPCompilingExec testQPCompilingExec.cpp QPCompilingExec.cpp QPIRPrepper.cpp ../Parse/RA/parser.cpp ../Parse/RA/scanner.cpp `llvm-config --cppflags --ldflags --libs` -w
