#!/bin/bash
clear
cd ../Parse/SQL
./compile.sh
cd ../../Analysis

# InsertWalker
g++ insertWalkerTest.cpp InsertWalker.cpp ../Parse/SQL/parser.cpp ../Parse/SQL/scanner.cpp ../../DataMgr/Metadata/Catalog.cpp -o insertWalkerTest -std=c++11 -w

# TypeChecker
g++ -o typeCheckerTest typeCheckerTest.cpp TypeChecker.cpp ../Parse/SQL/parser.cpp ../Parse/SQL/scanner.cpp ../../DataMgr/Metadata/Catalog.cpp -std=c++11 -w

# DdlWalker
g++ -o ddlWalkerTest ddlWalkerTest.cpp ../Parse/SQL/parser.cpp ../Parse/SQL/scanner.cpp ../../DataMgr/Metadata/Catalog.cpp -std=c++11 -w

# analysisTest
g++ -o analysisTest analysisTest.cpp Analysis.cpp InsertWalker.cpp TypeChecker.cpp ../Parse/SQL/parser.cpp ../Parse/SQL/scanner.cpp ../../DataMgr/Metadata/Catalog.cpp -std=c++11 -w

# Check the exit status
if [ "$?" != "0" ]; then
	echo "error"
fi
