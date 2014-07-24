// bufferTest.cpp

#include <iostream>
#include "Buffer.h"
#include "../../Shared/ansi.h"
#include "../../Shared/macros.h"
#include "../../Shared/testing.h"

using namespace Testing;
using namespace Buffer_Namespace;

#define BLOCKSIZE 32

// unit test function prototypes
void test_Buffer();
void test_Buffer_write();

int main() {
	PRINT_DLINE(80);

	test_Buffer();
	test_Buffer_write();

	PRINT_DLINE(80);
	printTestSummary();
	PRINT_DLINE(80);

	return EXIT_SUCCESS;
}

void test_Buffer() {
	mapd_size_t numPages = 32;
	mapd_size_t pageSize = 32;
	int *a = new int[numPages * pageSize];
	Buffer b((mapd_addr_t)a, numPages, pageSize);

	b.print();
	PRINT_SLINE(80);

	if (b.host_ptr() != (mapd_addr_t)a)
		PFAIL("incorrect host pointer address");
	else
		PPASS("correct host pointer address");

	if (b.length() != 0)
		PFAIL("length not initialized to 0");
	else
		PPASS("length initialized to 0");		

	if (!b.pinned())
		PFAIL("buffer not pinned (it should be pinned upon creation)");
	else
		PPASS("buffer pinned upon creation");

	if (b.dirty())
		PFAIL("buffer should not be dirty when created");
	else
		PPASS("buffer is clean (not dirty) upon creation");

	if (b.numPages() != numPages)
		PFAIL("incorrect number of pages when creating buffer");
	else
		PPASS("correct number of pages upon creation of buffer");

	if (b.pageSize() != pageSize)
		PFAIL("incorrect page size");
	else
		PPASS("correct page size");

	if (b.size() != (numPages * pageSize))
		PFAIL("incorrect buffer size");
	else
		PPASS("correct buffer size");

	b.unpin();
	if (b.pinned())
		PFAIL("unpinning the buffer failed");
	else
		PPASS("buffer unpinned successfully");

	b.pin();
	if (!b.pinned())
		PFAIL("pinning the buffer failed");
	else
		PPASS("pinning the buffer was a success -- all hail the ChunkKey Unicorn");
}

void test_Buffer_write() {
	bool err = false;

	// Create buffer
	mapd_size_t memPadFactor = 2;
	mapd_size_t numInt = 1024;
	mapd_size_t memSize = numInt * sizeof(int);
	mapd_size_t pageSize = 32;
	mapd_size_t numPages = memSize / pageSize;

	// Setup
	int a[numInt];
	Buffer b((mapd_addr_t)a, memPadFactor*numPages, pageSize);
	int *buf = (int*)b.host_ptr();
	int data[numInt];
	for (int i = 0; i < numInt; ++i)
		data[i] = i+1;

	// Create the buffer
	PRINT_SLINE(80);
	b.print();
	PRINT_SLINE(80);

	// Verify that the pointers match
	if (buf != a)
		PFAIL("pointer mismatch");
	else
		PPASS("pointers match");

	// Perform the write
	if (!b.write(0, memSize, (mapd_addr_t)data))
		PFAIL("write() method reported a failure");
	else
		PPASS("write() method succeeded");

	// Check that the buffer length has been updated
	if (b.length() != memSize)
		PFAIL("write() failed; buffer length mismatch");
	else
		PPASS("write() succeeded; buffer length correct");

	// Check that the data written is the same as the source data
	err = false;
	for (int i = 0; i < numInt; ++i) {
		if (buf[i] != data[i]) {
			err = true;
			break;
		}
	}
	if (err)
		PFAIL("write() failed; element mismatch");
	else
		PPASS("write() succeeded; no element mismatches");

	// Check dirty flags -- all pages should be dirty
	err = false;
	std::vector<bool> dirtyFlags = b.getDirty();
	for (int i = 0; i < dirtyFlags.size(); ++i) {
		if (!dirtyFlags[i]) {
			err = true;
			break;
		}
	}
	if (err)
		PFAIL("page incorrectly flagged as not dirty");
	else
		PPASS("all pages correctly flagged as not dirty.");
}




