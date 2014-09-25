bison++ -d -hparser.h -o parser.cpp parser.y
flex++ -d -i -oscanner.cpp scanner.l
g++ -O3 -o perfTest parser.cpp scanner.cpp tests/perfTest.cpp ../../../DataMgr/Metadata/Catalog.cpp ../../../DataMgr/PgConnector/PgConnector.cpp -w -I/usr/local/include -L/usr/local/lib -lboost_timer -lboost_system -lpqxx -std=c++11