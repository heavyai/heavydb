// bufferMgrTest.cpp

#include <iostream>
#include "BufferMgr.h"
#include "../File/FileMgr.h"

int main() {
	FileMgr fm(".");
	BufferMgr bm(1048576, NULL);

	return EXIT_SUCCESS;
}