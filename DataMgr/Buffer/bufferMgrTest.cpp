// bufferMgrTest.cpp

#include <iostream>
#include "BufferMgr.h"
#include "../File/FileMgr.h"
#include "../../Shared/ansi.h"
#include "../../Shared/testing.h"
#include "../../Shared/macros.h"

using namespace Testing;
using namespace Buffer_Namespace;
using namespace File_Namespace;

#define BLOCKSIZE 32

// unit test function prototypes
bool test_BufferMgr();

int main() {
	test_BufferMgr() ?
	    PPASS("BufferMgr()") : PFAIL("BufferMgr()");
	
	PRINT_DLINE(80);
	printTestSummary();
	PRINT_DLINE(80);
	return EXIT_SUCCESS;
}

bool test_BufferMgr() {
	BufferMgr(50 * 4096, NULL);
	return true;
}

