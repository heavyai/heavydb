/**
 * @file fileMgrTest.cpp
 * @author Steven Stewart <steve@map-d.com>
 *
 * This program is used to test the FileMgr class.
 *
 * @see FileMgr
 */
#include <cstdio>
#include <cstdlib>
#include "FileMgr.h"
#include "../../Shared/ansi.h"
#include "../../Shared/testing.h"

using namespace Testing;

// unit test function prototypes
// bool test_open();

int main(void) {
	FileMgr fm(".");
	fm.print();

	return EXIT_SUCCESS;
}

