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

// unit test function prototypes
void test_BufferMgr();
void test_createBuffer();
void test_createChunk(mapd_size_t numPages, mapd_size_t pageSize);
void test_getChunkBuffer(mapd_size_t numPages, mapd_size_t pageSize);
void test_flush();

int main() {
	test_BufferMgr();
	test_createBuffer();
	test_createChunk(32, 64);
	test_createChunk(1, 1);
	test_createChunk(300, 64);
	test_createChunk(3, 64);
	test_getChunkBuffer(32, 64);
	test_flush();

	return EXIT_SUCCESS;
}

void test_BufferMgr() {
	mapd_err_t err;

	// create a Buffer Mgr without a fileMgr object
	BufferMgr *bm = new BufferMgr(32 * 4096, NULL);
	if (bm == NULL)	
		PFAIL("BufferMgr didn't construct");
	else
		PPASS("BufferMgr constructed woo");

	FileMgr *fm = new FileMgr(".");
	bm = new BufferMgr(32 * 4096, fm);
	if (bm == NULL)	
		PFAIL("BufferMgr didn't construct w/fileMgr param");
	else
		PPASS("BufferMgr constructed with fileMgr param");
}

void test_createBuffer() {
	FileMgr *fm = new FileMgr(".");
	BufferMgr *bm = new BufferMgr(32 * 64, fm);

	Buffer *b = bm->createBuffer(32, 64);
	mapd_size_t freeMemSize = bm->freeMemSize();

	if (freeMemSize != 0) 
		PFAIL("create Buffer did not use up all memory");
	else
		PPASS("createBuffer split freeMem_ properly");

	// @todo when deleteChunk is written, write test for it 
}

void test_createChunk(mapd_size_t numPages, mapd_size_t pageSize) {
    FileMgr fm(".");
	BufferMgr *bm = new BufferMgr(numPages * pageSize, &fm);

	int keyInt = 4;
	ChunkKey key;
    key.push_back(keyInt);

	mapd_size_t srcBuf[pageSize*numPages];

	for (int i = 0; i < pageSize*numPages; i++) {
        srcBuf[i] = 42;
    }

    // create a Chunk from source address and make it into a buffer 
	Chunk* c = fm.createChunk(key, pageSize*numPages, pageSize, srcBuf, 0);
	Buffer *b = bm->createChunk(key, numPages, pageSize);

	if (b == NULL)	
		PFAIL("Null buffer returned from createChunk");
	else
		PPASS("Buffer returned is not NULL");

	if (bm->chunkIsCached(key))
		PFAIL("chunk ain't cached");
	else
		PPASS("cache rules everything around me");

	key.push_back(keyInt+1);

    // create a Chunk from source address and make it into a buffer 
	c = fm.createChunk(key, 0, pageSize, NULL, 0);
	b = bm->createChunk(key, numPages, pageSize);

	if (b == NULL)	
		PPASS("Null buffer returned from createChunk");
	else
		PFAIL("Buffer returned should be NULL but is not");

	//bm->printMemAlloc();
	bm->printChunkIndex();
}

void test_getChunkBuffer(mapd_size_t numPages, mapd_size_t pageSize) {
    FileMgr fm(".");
	BufferMgr *bm = new BufferMgr(numPages * pageSize, &fm);

	int keyInt = 4;
	ChunkKey key;
    key.push_back(keyInt);
	
    mapd_byte_t srcBuf[numPages*pageSize];
	mapd_byte_t destBuf[numPages*pageSize];

	for (int i = 0; i < numPages*pageSize; i++) {
        srcBuf[i] = 42;
    }
	for (int i = 0; i < numPages*pageSize; i++) {
        destBuf[i] = 43;
    }

    // create a Chunk and get it into a Buffer object
	Chunk* c = fm.createChunk(key, numPages*pageSize, pageSize, srcBuf, 0);

	Buffer *b = bm->getChunkBuffer(key);

	if (b == NULL)	
		PFAIL("Null buffer returned from createChunk");
	else
		PPASS("Buffer returned is not NULL");

	if (bm->chunkIsCached(key))
		PFAIL("chunk ain't cached");
	else
		PPASS("cache rules everything around me");

	if (!b->copy(0, numPages*pageSize, (mapd_addr_t)destBuf)) 
		PFAIL("Failed to copy to destBuf");
	else
		PPASS("Successfully copied to destBuf");


	bool loopError = false;
	for (int i = 0; i < numPages*pageSize; i++) {
        if (srcBuf[i] != destBuf[i]) {
        	loopError = true;
        }
    }

    if (loopError)
        PFAIL("destBuf filled during the copy from the buffer made by getChunkBuffer does not equal srcBuf created");
    else PPASS("srcBuf and destBuf match");

	if (b->copy(0, numPages*pageSize+1, (mapd_addr_t)destBuf)) 
		PFAIL("Somehow copied more bytes to buf than existed");
	else
		PPASS("Successfully exited upon improper input");

}

// grab your plunger cause it's time to flush
void test_flush() {
	mapd_size_t numPages = 64;
	mapd_size_t pageSize = 32;

    FileMgr fm(".");
	BufferMgr *bm = new BufferMgr(numPages * pageSize, &fm);

	int keyInt = 4;
	ChunkKey key;
    key.push_back(keyInt);
	
	printf("here yet?\n");
    mapd_byte_t srcBuf[numPages*pageSize];
	
	for (int i = 0; i < numPages*pageSize; i++) {
        srcBuf[i] = 42;
    }
	printf("here yet?\n");
    // create a Chunk and get it into a Buffer object
	Chunk* c = fm.createChunk(key, numPages*pageSize, pageSize, srcBuf, 1);
	Buffer *b = bm->getChunkBuffer(key);

	printf("here yet?\n");
	if (bm->flushChunk(key))
		PPASS("Successfully flushed buffer's contents to Chunk.");
	else
		PFAIL("Unsuccessfully flushed buffer's contents to Chunk");

	printf("here yet?\n");
	// upon flushing, the epoch should be set to 0 @todo fix epoch handling
	bool loopError = false;
	int epoch;
	printf("here yet?\n");
	(*c)[0]->current(&epoch);
	printf("current: %d\n", epoch);
	for (int i = 0; i < c->size(); i++) {
		if (epoch != 0)
			loopError = true;
	}

	if (loopError) 
		PFAIL("Epoch not set by flushChunk() correctly.");
	else
		PPASS("Epoch set by flushChunk() correctly");
}