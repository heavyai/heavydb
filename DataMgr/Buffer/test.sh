clear
g++ -g -o bufferTest Buffer.cpp bufferTest.cpp -std=c++11 -DDEBUG_VERBOSE
g++ -g -o bufferMgrTest BufferMgr.cpp Buffer.cpp bufferMgrTest.cpp ../File/File.cpp ../File/FileMgr.cpp -std=c++11 -DDEBUG_VERBOSE
./bufferTest
./bufferMgrTest

