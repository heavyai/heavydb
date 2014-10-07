clear
cd ../../Parse/SQL
bison++ -d -hparser.h -o parser.cpp parser.y
flex++ -d -i -oscanner.cpp scanner.l
cd ../../Plan/tests

g++ TranslatorTest.cpp ../Plan.cpp ../Translator.cpp ../../Parse/SQL/scanner.cpp ../../Parse/SQL/parser.cpp ../../Parse/RA/visitor/XMLTranslator.cpp ../../Parse/SQL/visitor/XMLTranslator.cpp ../../../DataMgr/PgConnector/PgConnector.cpp ../../../DataMgr/Metadata/Catalog.cpp  -I/usr/local/include -L/usr/local/lib/ -lpqxx -o translatorTest -w -std=c++11

g++ PlannerTest.cpp ../Planner.cpp ../Plan.cpp ../Translator.cpp ../../Parse/SQL/scanner.cpp ../../Parse/SQL/parser.cpp ../../Parse/RA/visitor/XMLTranslator.cpp ../../Parse/SQL/visitor/XMLTranslator.cpp ../../../DataMgr/PgConnector/PgConnector.cpp ../../../DataMgr/Metadata/Catalog.cpp  -I/usr/local/include -L/usr/local/lib/ -lpqxx -o plannerTest -w -std=c++11
