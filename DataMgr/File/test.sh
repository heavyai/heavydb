clear
g++ -g -o blockTest blockTest.cpp -std=c++11
g++ -o fileMgrTest FileMgr.cpp File.cpp fileMgrTest.cpp -std=c++11
g++ -o fileTest fileTest.cpp File.cpp -std=c++11
./blockTest
./fileTest
./fileMgrTest
rm *.mapd
