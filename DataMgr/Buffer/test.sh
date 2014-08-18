clear
g++ -g -o bufferTest Buffer.cpp bufferTest.cpp -std=c++11 -DDEBUG_VERBOSE
g++ -g -o bufferMgrTest BufferMgr.cpp Buffer.cpp bufferMgrTest.cpp ../PgConnector/PgConnector.cpp ../File/File.cpp ../File/FileMgr.cpp -std=c++11 -DDEBUG_VERBOSE -I/usr/local/include -L/usr/local/lib -lpqxx
./bufferTest
./bufferMgrTest
rm *.mapd
