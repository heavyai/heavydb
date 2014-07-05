/**
 * @file	FileMgr.cpp
 * @author	Steven Stewart <steve@map-d.com>
 *
 * Implementation file for the file managr.
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
     : fileId(fileId), f(f), blockSize(blockSize), nblocks(nblocks)
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

FileMgr::FileMgr(const std::string &basePath) {
	basePath_ = basePath;
	nextFileId_ = 0;
}

FileMgr::~FileMgr() {
	for (int i = 0; i < files_.size(); ++i)
		delete files_[i];
}

FileInfo* FileMgr::createFile(const mapd_size_t blockSize, const mapd_size_t nblocks, mapd_err_t *err) {
    FILE *f = NULL;

    // create the new file
    f = File::create(nextFileId_, blockSize, nblocks, err);

    // check for error
    if (!f || *err != MAPD_SUCCESS) {
    	fprintf(stderr, "[%s:%d] Error (%d): unable to create file.\n", __func__, __LINE__, *err);
    	return NULL;
    }

	// insert file into file manager data structures
	int fileId = nextFileId_++;

	// insert a new FileInfo object into vector of files
	FileInfo *fInfo = NULL;
	try {
		fInfo = new FileInfo(fileId, f, blockSize, nblocks);
		files_.push_back(fInfo);

		// insert fileId into multimap fileIndex (maps block sizes to files ids)
		fileIndex_.insert(std::pair<mapd_size_t, int>(blockSize, fileId));
	}
	catch (const std::bad_alloc& e) {
		std::cout << "Bad allocation exception encountered: " << e.what() << std::endl;
		*err = MAPD_FAILURE;
		return NULL;
	}
	catch (const std::exception& e) {
		*err = MAPD_FAILURE;
		std::cout << "Exception encountered: " << e.what() << std::endl;
		if (!fInfo) delete fInfo;
		return NULL;
	}
	assert(files_.back() == fInfo);
	return fInfo;
}

FileInfo* FileMgr::getFile(const int fileId, mapd_err_t *err) {
    if (fileId < 0 || fileId > files_.size()) {
    	*err = MAPD_ERR_FILE_NOT_FOUND;
    	return NULL;
    }
    return files_[fileId];
}

mapd_err_t FileMgr::deleteFile(const int fileId, const bool destroy) {
	mapd_err_t err = MAPD_SUCCESS;

    // confirm the file exists and obtain pointer
    FileInfo *fInfo = getFile(fileId, &err);
    if (err != MAPD_SUCCESS)
    	return err;

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

    return err;
}

BlockAddr* FileMgr::getBlock(const int fileId, mapd_size_t blockNum, mapd_err_t *err) {
	FileInfo *fInfo = FileMgr::getFile(fileId, err);
    return !fInfo ? NULL : getBlock(*fInfo, blockNum, err);
}

BlockAddr* FileMgr::getBlock(FileInfo &fInfo, mapd_size_t blockNum, mapd_err_t *err) {
    if (err) *err = MAPD_FAILURE;
    if (blockNum < fInfo.blocks.size()) {
        BlockAddr *bAddr = fInfo.blocks[blockNum];
        if (err && bAddr)
        	*err = MAPD_SUCCESS;
        return bAddr;
    }
    return NULL;
}

mapd_err_t FileMgr::clearBlock(const int fileId, mapd_size_t blockNum) {
	mapd_err_t err = MAPD_SUCCESS;
	FileInfo *fInfo = FileMgr::getFile(fileId, &err);
    return !fInfo ? err : clearBlock(*fInfo, blockNum);
}

mapd_err_t FileMgr::clearBlock(FileInfo &fInfo, mapd_size_t blockNum) {
    mapd_err_t err = MAPD_SUCCESS;
    BlockAddr *bAddr = getBlock(fInfo, blockNum, &err);
    if (bAddr && err == MAPD_SUCCESS)
    	bAddr->endByteOffset = 0;
    return err;
}

mapd_err_t FileMgr::freeBlock(const int fileId, mapd_size_t blockNum) {
    mapd_err_t err = MAPD_SUCCESS;
    FileInfo *fInfo = getFile(fileId, &err);
    return !fInfo ? err : freeBlock(*fInfo, blockNum);
}

mapd_err_t FileMgr::freeBlock(FileInfo &fInfo, mapd_size_t blockNum) {
    mapd_err_t err = MAPD_SUCCESS;
    err = clearBlock(fInfo, blockNum);
    if (err == MAPD_SUCCESS)
    	fInfo.freeBlocks.insert(blockNum);
    return err;
}

Chunk* FileMgr::getChunkRef(const ChunkKey &key, mapd_err_t *err) {
    ChunkKeyToChunkMap::iterator iter = chunkIndex_.find(key);
    if (iter != chunkIndex_.end()) {
        // found
        if (err) *err = MAPD_SUCCESS;
        return &iter->second;
    }
    else {
        // not found
        if (err) *err = MAPD_ERR_CHUNK_NOT_FOUND; // chunk doesn't exist
        return NULL;
    }
}

Chunk* FileMgr::getChunkCopy(const ChunkKey &key, mapd_byte_t *buf, mapd_err_t *err) {
    assert(buf);
    *err = MAPD_SUCCESS;
    
    ChunkKeyToChunkMap::iterator iter = chunkIndex_.find(key);
    if (iter == chunkIndex_.end()) {
        // not found
        *err = MAPD_ERR_CHUNK_NOT_FOUND; // chunk doesn't exist
        return NULL;
    }
    
    // found
    Chunk &c = iter->second;

    // copy contents of chunk to buf
    for (int i = 0; i < c.size(); ++i) {
        
        // get most recent address of current block
        BlockAddr *blk = c[i]->addr.back();

        // obtain a reference to the file of the block address
        int fileId = blk->fileId;
        FileInfo *fInfo = getFile(fileId, err);
        assert(fInfo);
        
        if (*err != MAPD_SUCCESS)
            return NULL;
        
        // open the file if it is not open already
        if (!fInfo->f)
            fInfo->f = File::open(fileId, err);
        if (*err != MAPD_SUCCESS)
            return NULL;
        
        // read block from file into buf
        File::read(fInfo->f, i * c[i]->blockSize, c[i]->blockSize, buf, err);
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
	Chunk *c = getChunkRef(key, &err);
	return !c ? err : clearChunk(*c);
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





