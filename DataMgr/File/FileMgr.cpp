/**
 * @file	FileMgr.cpp
 * @author	Steven Stewart <steve@map-d.com>
 *
 * Implementation file for the file manager.
 *
 * @see FileMgr.h
 */
#include <iostream>
#include <cassert>
#include <cstdio>
#include <string>
#include <cassert>
#include <exception>
#include "FileMgr.h"

using std::vector;

FileInfo::FileInfo(int fileId, FILE *f, mapd_size_t blockSize, mapd_size_t nblocks)
     : fileId(fileId), f(f), blockSize(blockSize), nblocks(nblocks) // STEVE: careful here - assignment to same variable name fails on some compilers even though it should work according to C++ standard
{
    // initialize blocks and free block list
    for (mapd_size_t i = 0; i < nblocks; ++i) {
        blocks.push_back(new BlockAddr(fileId, i * blockSize));
        freeBlocks.insert(i);
    }
    // @todo this assumes all are free blocks; how about reading a file header with free block info?
}

FileInfo::~FileInfo() {
	// free memory used by BlockAddr objects
	for (int i = 0; i < blocks.size(); ++i)
		delete blocks[i];

	// close file, if applicable
    if (f && File::close(f) != MAPD_SUCCESS)
        fprintf(stderr, "[%s:%d] Error closing file %d.\n", __func__, __LINE__, fileId);
}

void FileInfo::print(bool blockSummary) {
    printf("File #%d", fileId);
    printf(" size = %u", size());
    printf(" used = %u", used());
    printf(" free = %u", available());
    printf("\n");
    if (!blockSummary)
        return;
    
    for (int i = 0; i < blocks.size(); ++i) {
    	// @todo block summary
    }
}

FileMgr::FileMgr(const std::string &basePath): basePath_(basePath) {
	nextFileId_ = 0;
}

FileMgr::~FileMgr() {
	for (int i = 0; i < files_.size(); ++i)
		delete files_[i];

	// free memory allocated for Chunk objects
	for(auto it = chunkIndex_.begin(); it != chunkIndex_.end(); ++it) {
		Chunk &v = (*it).second;

		// free memory allocated for BlockInfo objects
		for (auto it2 = v.begin(); it2 != v.end(); ++it2)
			delete *it2;
	}
}

FileInfo* FileMgr::createFile(const mapd_size_t blockSize, const mapd_size_t nblocks) {
	if (blockSize < 1 || nblocks < 1)
		return NULL;

    // create the new file
    FILE *f = NULL;
    f = File::create(nextFileId_, blockSize, nblocks, NULL);

    // check for error
    if (!f) {
    	fprintf(stderr, "[%s:%d] Error: unable to create file.\n", __func__, __LINE__);
    	return NULL;
    }

	// update file manager data structures
    int fileId = nextFileId_++;
	FileInfo *fInfo = NULL;
	try {
		fInfo = new FileInfo(fileId, f, blockSize, nblocks);
		files_.push_back(fInfo);
		fileIndex_.insert(std::pair<mapd_size_t, int>(blockSize, fileId));
	}
	catch (const std::bad_alloc& e) {
		std::cout << "Bad allocation exception encountered: " << e.what() << std::endl;
		return NULL;
	}
	catch (const std::exception& e) {
		std::cout << "Exception encountered: " << e.what() << std::endl;
		if (!fInfo) delete fInfo;
		return NULL;
	}
	assert(files_.back() == fInfo);
	return fInfo;
}

FileInfo* FileMgr::getFile(const int fileId) {
    if (fileId < 0 || fileId > files_.size())
    	return NULL;
    return files_[fileId];
}

mapd_err_t FileMgr::deleteFile(const int fileId, const bool destroy) {
	// confirm the file exists and obtain pointer
    FileInfo *fInfo = getFile(fileId);
    if (!fInfo)
    	return MAPD_FAILURE;

    // remove the file from the fileIndex_
    BlockSizeFileMMap::iterator it = fileIndex_.lower_bound(fInfo->blockSize);
    for (it = fileIndex_.begin(); it != fileIndex_.end(); ++it) {
    	if (it->second == fileId)
    		break;
    }
    if (it != fileIndex_.end())
    	fileIndex_.erase(it);
    // @todo check else condition as error?

    // remove the file from the vector of files_
    files_.erase(files_.begin() + fileId);

    // @todo error-checking if erase fails?
    return MAPD_SUCCESS;
}

BlockAddr* FileMgr::getBlock(const int fileId, mapd_size_t blockNum) {
	FileInfo *fInfo = FileMgr::getFile(fileId);
    return !fInfo ? NULL : getBlock(*fInfo, blockNum);
}

BlockAddr* FileMgr::getBlock(FileInfo &fInfo, mapd_size_t blockNum) {
    assert(blockNum < fInfo.blocks.size() && fInfo.blocks[blockNum]);
    return fInfo.blocks[blockNum];
}

mapd_err_t FileMgr::putBlock(int fileId, mapd_size_t blockNum, mapd_size_t n, mapd_byte_t *buf) {
	FileInfo *fInfo;
	return ((fInfo = getFile(fileId)) == NULL) ? MAPD_FAILURE : putBlock(*fInfo, blockNum, n, buf);
}

mapd_err_t putBlock(FileInfo &fInfo, mapd_size_t blockNum, mapd_size_t n, mapd_byte_t *buf) {
	// The client should be writing blockSize bytes to the block
	assert(blockNum == fInfo.blockSize);

    // open the file if it is not open already
	mapd_err_t err;
	if (!fInfo.f) {
    	fInfo.f = File::open(fInfo.fileId, &err);
        if (err != MAPD_SUCCESS)
        	return err;
    }

    // write the block to the file
    size_t wrote = File::writeBlock(fInfo.f, fInfo.blockSize, blockNum, buf, &err);
    assert(wrote == fInfo.blockSize);

	return err;
}

mapd_err_t FileMgr::clearBlock(const int fileId, mapd_size_t blockNum) {
	FileInfo *fInfo = FileMgr::getFile(fileId);
    return !fInfo ? MAPD_FAILURE : clearBlock(*fInfo, blockNum);
}

mapd_err_t FileMgr::clearBlock(FileInfo &fInfo, mapd_size_t blockNum) {
    BlockAddr *bAddr = getBlock(fInfo, blockNum);
    if (bAddr) {
    	bAddr->endByteOffset = 0;
    	return MAPD_SUCCESS;
    }
    return MAPD_FAILURE;
}

mapd_err_t FileMgr::freeBlock(const int fileId, mapd_size_t blockNum) {
    FileInfo *fInfo = getFile(fileId);
    return !fInfo ? MAPD_FAILURE : freeBlock(*fInfo, blockNum);
}

mapd_err_t FileMgr::freeBlock(FileInfo &fInfo, mapd_size_t blockNum) {
    mapd_err_t err = MAPD_SUCCESS;
    err = clearBlock(fInfo, blockNum);
    if (err == MAPD_SUCCESS)
    	fInfo.freeBlocks.insert(blockNum); // @todo error-checking on insert() ?
    return err;
}

Chunk* FileMgr::getChunkRef(const ChunkKey &key) {
    auto it = chunkIndex_.find(key);
    return it != chunkIndex_.end() ? &it->second : NULL;
}

Chunk* FileMgr::getChunk(const ChunkKey &key, mapd_byte_t *buf) {
    assert(buf);
    
    // find chunk
    auto it = chunkIndex_.find(key);
    if (it == chunkIndex_.end()) // chunk doesn't exist
        return NULL;

    // copy contents of chunk to buf
    Chunk &c = it->second;
    for (int i = 0; i < c.size(); ++i) {

        // get most recent address of current block
        BlockAddr *blk = c[i]->addr.back();

        // obtain a reference to the file of the block address
        int fileId = blk->fileId;
        FileInfo *fInfo = getFile(fileId);
        if (!fInfo != MAPD_SUCCESS)
            return NULL;
        
        // open the file if it is not open already
        mapd_err_t err;
        if (!fInfo->f)
            fInfo->f = File::open(fileId, &err);
        if (err != MAPD_SUCCESS)
            return NULL;
        
        // read block from file into buf
        File::read(fInfo->f, i * c[i]->blockSize, c[i]->blockSize, buf, &err);
        if (err != MAPD_SUCCESS)
        	return NULL;
    }
    return &c;
}

mapd_err_t FileMgr::getChunkSize(const ChunkKey &key, int *nblocks, mapd_size_t *size) {
    assert(size || nblocks); // at least one of these should be not NULL
    mapd_err_t err = MAPD_SUCCESS;
    
    ChunkKeyToChunkMap::iterator iter = chunkIndex_.find(key);
    if (iter == chunkIndex_.end()) {
        // not found
        err = MAPD_ERR_CHUNK_NOT_FOUND; // chunk doesn't exist
        return err;
    }
    
    // found
    Chunk &c = iter->second;

    // check if chunk has no blocks
    if (nblocks) {
        *nblocks = c.size();
        if (*nblocks < 1) {
            if (size) *size = 0;
            return err;
        }
    }
    if (size) // Compute size based on block sizes
        for (int i = 0; i < c.size(); ++i)
            *size += c[i]->blockSize;

    return err;
}

mapd_err_t FileMgr::getChunkActualSize(const ChunkKey &key, mapd_size_t *size) {
    assert(size);
    mapd_err_t err = MAPD_SUCCESS;
    
    ChunkKeyToChunkMap::iterator iter = chunkIndex_.find(key);
    if (iter == chunkIndex_.end()) // not found
        return MAPD_ERR_CHUNK_NOT_FOUND;
    
    // Compute size based on actual bytes used in block
    Chunk &c = iter->second;
    for (int i = 0; i < c.size(); ++i)
        *size += c[i]->addr.back()->endByteOffset;
    
    return err;
}
/*
// Inserts free blocks into the chunk; creates a new file if necessary
Chunk* FileMgr::createChunk(ChunkKey &key, const mapd_size_t size, const mapd_size_t blockSize, void *src, mapd_err_t *err) {
    *err = MAPD_SUCCESS;
    
    // check if the chunk already exists based on key
    if (getChunkRef(key, NULL)) {
        if (err) *err = MAPD_ERR_CHUNK_DUPL;
        return NULL;
    }

    // instantiate, then determine number of blocks needed
    Chunk *c = new Chunk();
    mapd_size_t nblocks = (size + blockSize - 1) / blockSize;
    
    // obtain an iterator to files having the specified block size
    BlockSizeFileMMap::iterator it = fileIndex_.lower_bound(blockSize);
    int i = 0;
    for (; it != fileIndex_.end() && nblocks > 0; ++it) {
    	FileInfo *fInfo = getFile(it->second, err);

    	// for each free block in this file, give it to the new Chunk
    	vector<mapd_size_t> *free = &fInfo->freeBlocks;
    	for (int j = 0; j < free->size() && nblocks > 0; ++i, ++j, --nblocks) {
    		// remove block from free list
    		mapd_size_t blockNum = free->back();
    		free->pop_back();

    		// insert new BlockInfo into chunk
    		BlockInfo blk(blockSize, i);
    		blk.addr.push_back(BlockAddr(fInfo->fileId, blockNum*blockSize));
    		c->push_back(blk);

    		// write data into chunk on disk
    		if (src) {
    			// @todo update endByteOffset for each block of this chunk!
    			if (!fInfo->f)
    				fInfo->f = File::open(fInfo->fileId, err);
    			mapd_byte_t *srcAddr = (mapd_byte_t*)src + i*blockSize;
    			File::write(fInfo->f, blockNum*blockSize, blockSize, (void*)srcAddr, err);
    			// @todo clean this up and error-check
    		}
    	}
    }

    if (nblocks > 0) { // create a new file to hold remaining blocks
    	// @todo fix this
    	fprintf(stderr, "[%s:%d] Error: unable to insert %u blocks into new chunk.\n", __func__, __LINE__, nblocks);
    }
    
    return c;
}
*/
mapd_err_t FileMgr::clearChunk(ChunkKey &key) {
	// retrieve reference to the chunk
	mapd_err_t err = MAPD_SUCCESS;
	Chunk *c = getChunkRef(key);
	return !c ? MAPD_FAILURE : clearChunk(*c);
}

mapd_err_t FileMgr::clearChunk(Chunk &c) {
	for (int i = 0; i < c.size(); ++i)
		c[i]->addr.back()->clear();
	return MAPD_SUCCESS;
}

/*
mapd_err_t FileMgr::deleteChunk(Chunk &c) {
	// @todo go through each block and each copy of the block,
	// inserting them into the free lists of their respective files
	
    // Traverse the blocks of the chunk
    for (int i = 0; i < c.size(); i++) {
        //BlockInfo *bInfo = c[0];
    }

    return MAPD_SUCCESS;
}


void FileMgr::print() {
    
}

*/





