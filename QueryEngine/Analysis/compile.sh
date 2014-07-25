#!/bin/bash
clear
cd ../Parse/SQL
./compile.sh
cd ../../Analysis
g++ insertWalkerTest.cpp InsertWalker.cpp ../Parse/SQL/parser.cpp ../Parse/SQL/scanner.cpp ../../DataMgr/Metadata/Catalog.cpp -o insertWalkerTest -std=c++11 -w

# Check the exit status
if [ "$?" != "0" ]; then
	echo "error"
fi