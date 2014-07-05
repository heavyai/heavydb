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
#include <set>
#include "FileMgr.h"

using std::vector;

FileInfo::FileInfo(int fileId, FILE *f, mapd_size_t blockSize, mapd_size_t nblocks)
     : fileId(fileId), f(f), blockSize(blockSize), nblocks(nblocks)
{
    assert(f);
    
    // initialize blocks and free block list
    for (mapd_size_t i = 0; i < nblocks; ++i) {
        blocks.push_back(BlockAddr(fileId, i * blockSize));
        freeBlocks.insert(i);
    }
}

FileInfo::~FileInfo() {
    mapd_err_t err = MAPD_SUCCESS;
    if (f) err = File::close(f);
    if (err != MAPD_SUCCESS)
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
    
    /*for (int i = 0; i < blocks.size(); ++i) {

    }*/
}

FileMgr::FileMgr(const std::string &basePath) {
	basePath_ = basePath;
	nextFileId_ = 0;
}

FileMgr::~FileMgr() {
	for (int i = 0; i < files_.size(); ++i)
		delete &files_[i];
}

FileInfo* FileMgr::createFile(const mapd_size_t blockSize, const mapd_size_t nblocks, mapd_err_t *err) {
    FILE *f = NULL;
    FileInfo *fInfo = NULL;

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
	files_.push_back(FileInfo(fileId, f, blockSize, nblocks));

	// insert fileId into multimap fileIndex (maps block sizes to files ids)
	fileIndex_.insert(std::pair<mapd_size_t, int>(blockSize, fileId));

	return &files_.back();
}

FileInfo* FileMgr::getFile(const int fileId, mapd_err_t *err) {
    if (fileId < 0 || fileId > files_.size()) {
    	*err = MAPD_ERR_FILE_NOT_FOUND;
    	return NULL;
    }
    return &files_[fileId];
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
        BlockAddr *bAddr = &fInfo.blocks[blockNum];
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

// Note: The set is sorted, and we expect the most recent version of the block to be the
// last element. To clear it, we copy the last element, erase it, then reinsert, following
// the idiom described here:
// http://stackoverflow.com/questions/2217878/c-stl-set-update-is-tedious-i-cant-change-an-element-in-place
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

Chunk* FileMgr::getChunkCopy(const ChunkKey &key, void *buf, mapd_err_t *err) {
    assert(buf);
    *err = MAPD_SUCCESS;
    
    ChunkKeyToChunkMap::iterator iter = chunkIndex_.find(key);
    if (iter == chunkIndex_.end()) {
        // not found
        *err = MAPD_ERR_CHUNK_NOT_FOUND; // chunk doesn't exist
        return NULL;
    }
    
    // found
    Chunk *c = &iter->second;

    // copy contents of chunk to buf
    Chunk::iterator chunkIt = c->begin();
    for (int i = 0; chunkIt != c->end(); ++chunkIt, ++i) {
        BlockInfo bInfo = *chunkIt;
        
        // get most recent copy of block
        std::set<BlockAddr>::reverse_iterator it = bInfo.addr.rbegin();
        BlockAddr blk = *it;

        // determine which file the block belongs to
        int fileId = blk.fileId;
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
        File::read(fInfo->f, i * bInfo.blockSize, bInfo.blockSize, buf, err);
    }
    return c;
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
    Chunk *c = &iter->second;

    // check if chunk has no blocks
    if (nblocks) {
        *nblocks = c->size();
        if (*nblocks < 1) {
            if (size) *size = 0;
            return err;
        }
    }
    if (size) {
        // Compute size based on block sizes
        Chunk::iterator chunkIt = c->begin();
        for (int i = 0; chunkIt != c->end(); ++chunkIt, ++i) {
            BlockInfo bInfo = *chunkIt;
            *size += bInfo.blockSize;
        } // @todo This could be sped a little if we assume each block of a chunk is the same size, which is likely true
    }
    return err;
}

mapd_err_t FileMgr::getChunkActualSize(const ChunkKey &key, mapd_size_t *size) {
    assert(size);
    mapd_err_t err = MAPD_SUCCESS;
    
    ChunkKeyToChunkMap::iterator iter = chunkIndex_.find(key);
    if (iter == chunkIndex_.end()) {
        // not found
        err = MAPD_ERR_CHUNK_NOT_FOUND; // chunk doesn't exist
        return err;
    }
    
    // found
    Chunk *c = &iter->second;
    
    // Compute size based on actual bytes used in block
    Chunk::iterator chunkIt = c->begin();
    for (int i = 0; chunkIt != c->end(); ++chunkIt, ++i) {
        BlockInfo bInfo = *chunkIt;

        // get most recent copy of block
        std::set<BlockAddr>::reverse_iterator it = bInfo.addr.rbegin();
        BlockAddr blk = *it;

        *size += blk.endByteOffset;
    } // @todo This could be sped a little if we assume each block of a chunk is the same size, which is likely true
    
    return err;
}
/*
Chunk* FileMgr::createChunk(ChunkKey &key, const mapd_size_t size, const mapd_size_t blockSize, const void *src, mapd_err_t *err) {
    *err = MAPD_SUCCESS;
    Chunk *c = NULL;

    // check if the chunk already exists
    if (getChunkRef(key, NULL)) {
        if (err) *err = MAPD_ERR_CHUNK_DUPL;
        return NULL;
    }

    // determine number of blocks needed
    mapd_size_t nblocks = (size + blockSize - 1) / blockSize;
    
    // obtain an iterator to files that use the specified block size
    BlockSizeFileMMap::iterator it = fileIndex_.lower_bound(blockSize);
    for (; it != fileIndex_.end(); ++it) {
    	// @todo
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

mapd_err_t clearChunk(Chunk &c) {
	std::set<BlockInfo>::iterator it;
	for (it = c.begin(); it != c.end(); ++it) {
		std::set<BlockAddr> *pSet = &(it->addr);
		std::set<BlockAddr>::reverse_iterator it = pSet->rbegin();
		BlockAddr copy = *it;
		copy.endByteOffset = 0;
		pSet->erase(*it);
		pSet->insert(copy);
	}
	return MAPD_SUCCESS;
}


mapd_err_t FileMgr::deleteChunk(Chunk &c) {
	// @todo go through each block and each copy of the block,
	// inserting them into the free lists of their respective files
	return MAPD_SUCCESS;
}

/*
void FileMgr::print() {
    
}

*/





