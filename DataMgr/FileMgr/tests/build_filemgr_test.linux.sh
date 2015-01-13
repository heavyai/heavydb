g++ -O3 --std=c++0x -o FileMgrTest FileMgrTest.cpp ../FileMgr.cpp ../File.cpp ../FileBuffer.cpp ../FileInfo.cpp  -I/usr/local/include -lgtest -lboost_filesystem -lboost_system -lboost_timer -pthread
