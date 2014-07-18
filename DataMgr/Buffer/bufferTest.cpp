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
bool test_Buffer();
bool test_Buffer_write();

int main() {
	PRINT_DLINE(80);

	test_Buffer() ?
	    PPASS("Buffer::Buffer()") : PFAIL("Buffer::Buffer()");
	test_Buffer_write() ?
	    PPASS("Buffer::write()") : PFAIL("Buffer::write()");

	PRINT_DLINE(80);
	printTestSummary();
	PRINT_DLINE(80);

	return EXIT_SUCCESS;
}

bool test_Buffer() {
	mapd_size_t numPages = 32;
	mapd_size_t pageSize = 32;
	int *a = new int[numPages * pageSize];
	Buffer b((mapd_addr_t)a, numPages, pageSize);

	b.print();
	PRINT_SLINE(80);

	if (b.host_ptr() != (mapd_addr_t)a) {
		fprintf(stderr, "[%s:%d] Error: incorrect host pointer address (%p != %p).\n", __func__, __LINE__, b.host_ptr(), a);
		return false;		
	}

	if (b.length() != 0) {
		fprintf(stderr, "[%s:%d] Error: length not initialized to 0.\n", __func__, __LINE__);
		return false;
	}

	if (!b.pinned()) {
		fprintf(stderr, "[%s:%d] Error: buffer not pinned (it should be pinned upon creation).\n", __func__, __LINE__);
		return false;
	}

	if (b.dirty()) {
		fprintf(stderr, "[%s:%d] Error: buffer should not be dirty.\n", __func__, __LINE__);
		return false;
	}

	if (b.numPages() != numPages) {
		fprintf(stderr, "[%s:%d] Error: incorrect number of pages (%lu != %lu).\n", __func__, __LINE__, b.numPages(), numPages);
		return false;
	}

	if (b.pageSize() != pageSize) {
		fprintf(stderr, "[%s:%d] Error: incorrect page size (%lu != %lu).\n", __func__, __LINE__, b.numPages(), numPages);
		return false;
	}

	if (b.size() != (numPages * pageSize)) {
		fprintf(stderr, "[%s:%d] Error: incorrect buffer size (%lu != %lu).\n", __func__, __LINE__, b.size(), numPages*pageSize);
		return false;
	}

	b.unpin();
	if (b.pinned()) {
		fprintf(stderr, "[%s:%d] Error: unpinning the buffer failed.\n", __func__, __LINE__);
		return false;
	}

	b.pin();
	if (!b.pinned()) {
		fprintf(stderr, "[%s:%d] Error: pinning the buffer failed.\n", __func__, __LINE__);
		return false;
	}

	return true;
}

bool test_Buffer_write() {
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
	if (buf != a) {
		fprintf(stderr, "[%s:%d] Error: pointer mismatch: %p != %p.\n", __func__, __LINE__, buf, a);
		return false;
	}

	// Perform the write
	if (!b.write(0, memSize, (mapd_addr_t)data)) {
		fprintf(stderr, "[%s:%d] Error: write() method reported a failure.\n", __func__, __LINE__);
		return false;
	}

	// Check that the buffer length has been updated
	if (b.length() != memSize) {
		fprintf(stderr, "[%s:%d] Error: write() failed; buffer length mismatch: %lu != %lu\n",
			__func__, __LINE__, b.length(), (numPages*pageSize));
		return false;
	}

	// Check that the data written is the same as the source data
	for (int i = 0; i < numInt; ++i) {
		if (buf[i] != data[i]) {
			fprintf(stderr, "[%s:%d] Error: write() failed; element mismatch at position %d: %d != %d\n",
				__func__, __LINE__, i, buf[i], data[i]);
			return false;
		}
	}

	// Check dirty flags -- all pages should be dirty
	std::vector<bool> dirtyFlags = b.getDirty();
	for (int i = 0; i < dirtyFlags.size(); ++i) {
		if (!dirtyFlags[i]) {
			fprintf(stderr, "[%s:%d] Error: page #%d incorrectly flagged as not dirty.\n",
				__func__, __LINE__, i);
			return false;
		}
	}

	return true;
}




