clear
g++ -g -o blockTest blockTest.cpp -std=c++11
g++ -o fileMgrTest FileMgr.cpp File.cpp fileMgrTest.cpp -std=c++11
if [ "$?" != "0" ]; then
echo "l0l h4xed"
exit 0 
fi
g++ -o fileTest fileTest.cpp File.cpp -std=c++11

#./blockTest
#./fileTest
./fileMgrTest
rm *.mapd
