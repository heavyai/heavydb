// bufferMgrTest.cpp

#include <iostream>
#include "BufferMgr.h"
//#include "../File/FileMgr.h"
#include "../../Shared/ansi.h"
#include "../../Shared/testing.h"

using namespace Testing;
using namespace Buffer_Namespace;

#define BLOCKSIZE 32

// unit test function prototypes
bool test_BufferMgr();

int main() {
	test_BufferMgr() ?
	    PPASS("BufferMgr()") : PFAIL("BufferMgr()");

	return EXIT_SUCCESS;
}

bool test_BufferMgr() {
	//::FileMgr fm(".");
	BufferMgr bm1(1048576, NULL);
	BufferMgr bm2(1048576, 256, NULL);
	return true;
}
