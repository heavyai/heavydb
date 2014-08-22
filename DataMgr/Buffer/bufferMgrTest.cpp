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
void test_BufferMgr(mapd_size_t hostMemSize);
void test_createBuffer(mapd_size_t numPages, mapd_size_t pageSize);
void test_createChunk(mapd_size_t numPages, mapd_size_t pageSize);
void test_deleteChunk(mapd_size_t numPages, mapd_size_t pageSize, mapd_size_t bytesToWrite);
void test_getChunkBuffer(mapd_size_t numPages, mapd_size_t pageSize);
void test_flush(mapd_size_t numPages, mapd_size_t pageSize);
void test_getChunkAddr(mapd_size_t numPages, mapd_size_t pageSize);

int main() {
	test_BufferMgr(1);
	test_BufferMgr(10);
	test_BufferMgr(32*4096);
	
	PRINT_SLINE(70);
    
	test_createBuffer(1, 1);
	test_createBuffer(32, 32);
	test_createBuffer(1024, 1024);
	test_createBuffer(1, 65536);
	test_createBuffer(65536, 1);
    
	PRINT_SLINE(70);
    
	test_createChunk(1, 1);
	test_createChunk(32, 64);
	test_createChunk(1024, 1024);
	test_createChunk(1, 65536);
	test_createChunk(256, 1);
    
	PRINT_SLINE(70);
    
	test_deleteChunk(1, 1, 1);
	test_deleteChunk(32, 64, 32*64);
	test_deleteChunk(32, 64, 32*64/2);
	test_deleteChunk(1024, 1024, 1024*1024);
	test_deleteChunk(1, 65536, 65536);
	test_deleteChunk(1, 65536, 65536/2);
	test_deleteChunk(256, 1, 256);
	test_deleteChunk(256, 1, 128);
    
	PRINT_SLINE(70);
    
	test_getChunkBuffer(1, 1);
	test_getChunkBuffer(32, 64);
	test_getChunkBuffer(1024, 1024);
	test_getChunkBuffer(1, 65536);
	test_getChunkBuffer(16384, 1);
    
	PRINT_SLINE(70);
    
	test_flush(1, 1);
	test_flush(32, 64);
	test_flush(1024, 1024);
	test_flush(1, 65536);
	test_flush(4096, 1);
    
	PRINT_SLINE(70);
    
	test_getChunkAddr(1, 1);
	test_getChunkAddr(32, 64);
	test_getChunkAddr(1024, 1024);
	test_getChunkAddr(1, 65536);
	test_getChunkAddr(4096, 1);
	
	PRINT_DLINE(70);
	
	printTestSummary();
    
	PRINT_DLINE(70);
    
	return EXIT_SUCCESS;
}

void test_BufferMgr(mapd_size_t hostMemSize) {
	mapd_err_t err;
    
	// create a Buffer Mgr without a fileMgr object
	BufferMgr *bm = new BufferMgr(hostMemSize, NULL);
	if (bm == NULL)
		PFAIL("BufferMgr didn't construct");
	else
		PPASS("BufferMgr constructed woo");
    
	FileMgr *fm = new FileMgr(".");
	bm = new BufferMgr(hostMemSize, fm);
	if (bm == NULL)
		PFAIL("BufferMgr didn't construct w/fileMgr param");
	else
		PPASS("BufferMgr constructed with fileMgr param");
}

void test_createBuffer(mapd_size_t numPages, mapd_size_t pageSize) {
	FileMgr *fm = new FileMgr(".");
	BufferMgr *bm = new BufferMgr(numPages * pageSize, fm);
    
	Buffer *b = bm->createBuffer(numPages, pageSize);
	mapd_size_t freeMemSize = bm->unused();
    
	if (freeMemSize != 0)
		PFAIL("create Buffer did not use up all memory");
	else
		PPASS("createBuffer split freeMem_ properly");
    
	// @todo when deleteChunk is written, write test for it
	bm->deleteBuffer(b);
}

void test_createChunk(mapd_size_t numPages, mapd_size_t pageSize) {
    
    FileMgr fm(".");
	BufferMgr *bm = new BufferMgr(numPages * pageSize, &fm);
    
	int keyInt = 4;
	ChunkKey key;
    key.push_back(keyInt);
    
	mapd_byte_t *srcBuf = new mapd_byte_t[pageSize*numPages];
    
	for (int i = 0; i < pageSize*numPages; i++) {
		//printf("[%d] no pun int", i);
        srcBuf[i] = 42;
    }
    
    // create a Chunk from source address and make it into a buffer
	Chunk* c = fm.createChunk(key, pageSize*numPages, pageSize, srcBuf, 0);
	Buffer *b = bm->createChunk(key, numPages, pageSize);
    
	if (b == NULL)
		PFAIL("Null buffer returned from createChunk");
	else
		PPASS("Buffer returned is not NULL");
    
	if (!bm->chunkIsCached(key))
		PFAIL("chunk ain't cached");
	else
		PPASS("cache rules everything around me");
    
	key.push_back(keyInt+1);
    
	bm->deleteBuffer(b);
	b = NULL;
    
    // create a Chunk from source address and make it into a buffer
	c = fm.createChunk(key, 0, pageSize, NULL, 0);
	b = bm->createChunk(key, numPages, pageSize);
    
	if (b != NULL)
		PPASS("Buffer returned from createChunk on blank chunk");
	else
		PFAIL("No buffer returned from createChunk on blank chunk");
    
	//bm->printMemAlloc();
	bm->printChunkIndex();
	
	//bm->deleteBuffer(b);
	delete [] srcBuf;
}

void test_deleteChunk(mapd_size_t numPages, mapd_size_t pageSize, mapd_size_t bytesToWrite) {
    FileMgr fm(".");
	BufferMgr *bm = new BufferMgr(numPages * pageSize, &fm);
    
	int keyInt = 4;
	ChunkKey key;
    key.push_back(keyInt);
	
    mapd_byte_t *srcBuf = new mapd_byte_t[numPages*pageSize];
	mapd_byte_t *destBuf = new mapd_byte_t[numPages*pageSize];
    
	for (int i = 0; i < bytesToWrite; i++) {
        srcBuf[i] = 42;
    }
	for (int i = 0; i < bytesToWrite; i++) {
        destBuf[i] = 43;
    }
    
    // create a Chunk and get it into a Buffer object
	Chunk* c = fm.createChunk(key, bytesToWrite, pageSize, srcBuf, 0);
    
	int availableBegin = bm->unused();
    //	printf("used by Chunk's buffer: %d, unused: %d, total: %d\n", 0, bm->unused(), numPages*pageSize);
	
	Buffer *b = bm->getChunkBuffer(key);
	int availablePreFree = bm->unused();
	int usedByChunk = availableBegin - availablePreFree;
	bm->deleteChunk(key);
    
	int availablePostFree = bm->unused();
	//printf("used by Chunk's buffer: %d, unused: %d, total: %d\n", usedByChunk, bm->unused(), numPages*pageSize);
	
	if (availableBegin != availablePostFree)
		PFAIL("free mem size after deleteChunk() does not equal free mem size before createChunk()");
	else
		PPASS("free mem size after deleteChunk() equals free mem size before createChunk()");
    
	if (usedByChunk != availablePostFree - availablePreFree)
		PFAIL("# of bytes created in buffer does not equal number of bytes freed after deleteChunk()");
	else
		PPASS("# of bytes written to chunk equals created # of bytes in buffer");
    
	if (fm.getChunkRef(key) != NULL)
		PFAIL("getChunkRef() returning non-null reference to Chunk");
	else
		PPASS("getChunkRef returns NULL reference to Chunk");
    
}

void test_getChunkBuffer(mapd_size_t numPages, mapd_size_t pageSize) {
    FileMgr fm(".");
	BufferMgr *bm = new BufferMgr(numPages * pageSize, &fm);
    
	int keyInt = 4;
	ChunkKey key;
    key.push_back(keyInt);
	
    mapd_byte_t *srcBuf = new mapd_byte_t[numPages*pageSize];
	mapd_byte_t *destBuf = new mapd_byte_t[numPages*pageSize];
    
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
    
	if (!bm->chunkIsCached(key))
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
    
	bm->deleteBuffer(b);
	delete [] srcBuf;
	delete [] destBuf;
}

// grab your plunger cause it's time to flush
void test_flush(mapd_size_t numPages, mapd_size_t pageSize) {
	
    FileMgr fm(".");
	BufferMgr *bm = new BufferMgr(numPages * pageSize * 3, &fm);
    
    int keyInt[] = {1, 2, 3, 4, 5};
	ChunkKey key;
    key.push_back(keyInt[0]);
	
    mapd_byte_t *srcBuf = new mapd_byte_t[numPages*pageSize];
	
	for (int i = 0; i < numPages*pageSize; i++) {
        srcBuf[i] = 42;
    }
    // create a Chunk and get it into a Buffer object
	Chunk* c = fm.createChunk(key, numPages*pageSize, pageSize, srcBuf, 1);
	Buffer *b = bm->getChunkBuffer(key);
    
	// unpin buffer
	b->unpin();
	if (bm->flushChunk(key))
		PPASS("Successfully flushed buffer's contents to Chunk.");
	else
		PFAIL("Unsuccessfully flushed buffer's contents to Chunk");
    
	// upon flushing, the epoch should be set to 0 @todo fix epoch handling
	bool loopError = false;
	int epoch = -1 ;
	(*c)[0]->current(&epoch);
	//printf("current: %d\n", epoch);
	for (int i = 0; i < c->size(); i++) {
		if (epoch != 0)
			loopError = true;
	}
    
	if (loopError)
		PFAIL("Epoch not set by flushChunk() correctly.");
	else
		PPASS("Epoch set by flushChunk() correctly");
    
	// try flushing a pinned buffer
	b->pin();
	if (bm->flushChunk(key))
		PFAIL("Tried to flush pinned buffer's contents to Chunk.");
	else
		PPASS("Unable to flush pinned buffer's contents to Chunk");
    
	// make two more chunks and flush them
	ChunkKey key2;
	key2.push_back(keyInt[1]);
    
	// create and unpin buffer for flushing
	Chunk *c1 = fm.createChunk(key2, numPages*pageSize, pageSize, srcBuf, 1);
	b = bm->getChunkBuffer(key2);
	b->unpin();
    
	ChunkKey key3;
	key3.push_back(keyInt[2]);
    
	Chunk *c2 = fm.createChunk(key3, numPages*pageSize, pageSize, srcBuf, 1);
	b = bm->getChunkBuffer(key3);
	b->unpin();
    
	bm->flushAllChunks();
    
	// check the epochs to see if the flush worked
	loopError = false;
	(*c1)[0]->current(&epoch);
	for (int i = 0; i < c->size(); i++) {
		if (epoch != 0)
			loopError = true;
	}
	
	(*c2)[0]->current(&epoch);
	for (int i = 0; i < c->size(); i++) {
		if (epoch != 0)
			loopError = true;
	}
    
	if (loopError)
		PFAIL("Epoch not set by flushAllChunk() correctly.");
	else
		PPASS("Epoch set by flushAllChunk() correctly");
    
	bm->deleteBuffer(b);
	delete [] srcBuf;
}

void test_getChunkAddr(mapd_size_t numPages, mapd_size_t pageSize) {
    FileMgr fm(".");
	BufferMgr *bm = new BufferMgr(numPages * pageSize * 3, &fm);
    
    int keyInt[] = {1, 2, 3, 4, 5};
	ChunkKey key;
    key.push_back(keyInt[0]);
    
    mapd_byte_t *srcBuf = new mapd_byte_t[numPages*pageSize];
	
	for (int i = 0; i < numPages*pageSize; i++) {
        srcBuf[i] = 42;
    }
    // create a Chunk and get it into a Buffer object
	Chunk* c = fm.createChunk(key, numPages*pageSize, pageSize, srcBuf, 1);
	Buffer *b = bm->getChunkBuffer(key);
    
	mapd_size_t newLength = b->length();
	mapd_size_t tempLength;
    
	b->unpin();
	if (bm->getChunkAddr(key) == NULL)
		PFAIL("getChunkAddr returning NULL host pointer.");
	else
		PPASS("getChunkAddr returns non-Null host pointer.");
    
	b->unpin();
    
	bm->getChunkAddr(key, &tempLength);
    
	if (newLength != tempLength) 
		PFAIL("getChunkAddr not setting Length correctly");
	else
		PPASS("getChunkAddr sets Length correctly");
    
}
