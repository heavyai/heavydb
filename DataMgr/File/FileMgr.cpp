/**
 * @file	FileMgr.cpp
 * @author	Steven Stewart <steve@map-d.com>
 *
 * Implementation file for the file managr.
 *
 * @see FileMgr.h
 */
#include <iostream>
#include <cstdio>
#include <string>
#include <cassert>
#include "FileMgr.h"

unsigned int FileMgr::nextFileId = 0;

FileInfo::FileInfo(int fileId, FILE *f, mapd_size_t blockSize, mapd_size_t nblocks)
     : fileId(fileId), f(f), blockSize(blockSize), nblocks(nblocks)
{
    assert(f);
    
    // initialize blocks and free block list
    for (int i = 0; i < nblocks; ++i) {
        blocks.push_back(BlockInfo(BlockAddr(fileId, i * blockSize), blockSize, 0));
        freeBlocks.insert(i);
    }
}

FileInfo::~FileInfo() {
    mapd_err_t err = File::close(f);
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
    
    for (int i = 0; i < blocks.size(); ++i) {

        // obtain reference to current block
        BlockInfo &bInfo = blocks[i];

        // check if current block is in free list
        std::set<mapd_size_t>::iterator it = find(freeBlocks.begin(), freeBlocks.end(), i);
        bool isFree = it != freeBlocks.end();

        // print block metadata
        printf("[%d] %s%s %u/%u", i, (isFree ? "f":"-"), (bInfo.isShadow) ? "s" : "-", bInfo.endByteOffset, bInfo.blockSize);
        assert(bInfo.blockSize == this->blockSize);
        printf(" ep=%u", bInfo.epoch);
        printf("\n");
    }
}

FileMgr::FileMgr(const std::string &basePath) {
	basePath_ = basePath;
}

FileMgr::~FileMgr() {
    // NOP
    
    // @todo Free FileInfo* objects in FileMap
}

FileInfo* FileMgr::createFile(const mapd_size_t blockSize, const mapd_size_t nblocks, mapd_err_t *err) {
    FILE *f = NULL;
    FileInfo *fInfo = NULL;
    
    // create the new file
    f = File::create(nextFileId, blockSize, nblocks, err);
    
    if (f) {
        // insert a new FileInfo object into the FileMap
        fInfo = new FileInfo(nextFileId++, f, blockSize, nblocks);
        files_.insert(std::pair<int,FileInfo*>(fInfo->fileId, fInfo));
    }
    
    return fInfo;
}

mapd_err_t FileMgr::deleteFile(const int fileId) {
    mapd_err_t err;
    FileMap::iterator iter = files_.find(fileId);

    if (iter != files_.end()) { // delete FileInfo object
        // obtain FileInfo pointer
        FileInfo *fInfo = getFile(fileId, NULL);
        
        // free memory used by FileInfo object
        delete fInfo;
        fInfo = NULL;
        
        // remove FileMap entry
        files_.erase(iter);
        return MAPD_SUCCESS;
    }
    return MAPD_FAILURE;
}

FileInfo* FileMgr::getFile(const int fileId, mapd_err_t *err) {
    FileMap::iterator iter = files_.find(fileId);
    if (iter != files_.end()) {
        if (err) *err = MAPD_SUCCESS;
        return iter->second;
    }
    if (err) *err = MAPD_ERR_FILE_NOT_FOUND;
    return NULL;
}

BlockInfo* FileMgr::getBlock(const int fileId, mapd_size_t blockNum, mapd_err_t *err) {
    if (err) *err = MAPD_FAILURE;
    FileInfo *fInfo = FileMgr::getFile(fileId, err);
    if (fInfo && blockNum < fInfo->blocks.size()) {
        BlockInfo *bInfo = &fInfo->blocks[blockNum];
        if (err && bInfo) *err = MAPD_SUCCESS;
        return bInfo;
    }
    return NULL;
}

BlockInfo* getBlock(FileInfo &fInfo, mapd_size_t blockNum, mapd_err_t *err) {
    if (err) *err = MAPD_FAILURE;
    if (blockNum < fInfo.blocks.size()) {
        BlockInfo *bInfo = &fInfo.blocks[blockNum];
        if (err && bInfo) *err = MAPD_SUCCESS;
        return bInfo;
    }
    return NULL;
}

mapd_err_t FileMgr::clearBlock(const int fileId, mapd_size_t blockNum) {
    mapd_err_t err = MAPD_SUCCESS;
    BlockInfo *bInfo = getBlock(fileId, blockNum, &err);
    if (bInfo && err == MAPD_SUCCESS)
        bInfo->endByteOffset = 0;
    return err;
}

mapd_err_t FileMgr::freeBlock(const int fileId, mapd_size_t blockNum) {
    mapd_err_t err = MAPD_SUCCESS;
    FileInfo *fInfo = getFile(fileId, &err);
    fInfo->freeBlocks.insert(blockNum);
    return err;
}

/**
 * @brief Finds the chunk using the key, and returns the reference in c.
 *
 * The chunk is found using the key, which is passed to the find() method of chunkIndex_, 
 * which is a map from ChunkKey to Chunk. If found, a pointer to the chunk is returned and
 * err is set to MAPD_SUCCESS; otherwise, NULL is returned and err is set to
 * MAPD_ERR_CHUNK_NOT_FOUND.
 * 
 */
/*Chunk* FileMgr::getChunkRef(const ChunkKey &key, mapd_err_t *err) {
    ChunkKeyToChunkMap::iterator iter = chunkIndex_.find(key);
    if (iter != chunkIndex_.end()) {
        // found
        *err = MAPD_SUCCESS;
        return &iter->second;
    }
    else {
        // not found
        *err = MAPD_ERR_CHUNK_NOT_FOUND; // chunk doesn't exist
        return NULL;
    }
}*/

/**
 * @brief Finds the chunk using the key, and then copies it to buf.
 *
 * The chunk is found using the key, which is passed to the find() method of chunkIndex_, 
 * which is a map from ChunkKey to Chunk. If found, the contents of the chunk is copied to
 * the location pointed to by buf, a pointer to the Chunk object is returned, and err is set
 * to MAPD_SUCCESS.
 *
 */
/*Chunk* FileMgr::getChunkCopy(const ChunkKey &key, void *buf, mapd_err_t *err) {
    Chunk *c = getChunkRef(key, err);
    if (*err != MAPD_SUCCESS)
        return NULL;
    
    // copy contents of chunk to buf
    Chunk::iterator iter;
    for (iter = c->begin(); iter != c->end(); ++iter) {
        BlockInfo binfo = *iter;
        File f(binfo.blockSize);
        f.open(binfo.blk.fileId);
    }
    
    return c;
}*/

/*
mapd_err_t FileMgr::getChunkSize(const ChunkKey &key, int *nblocks, mapd_size_t *size) const {
    
}

mapd_err_t FileMgr::createChunk(ChunkKey &key, const mapd_size_t requested, mapd_size_t *actual, const void *src) {
    
}

mapd_err_t FileMgr::deleteChunk(const ChunkKey &key, mapd_size_t *nblocks, mapd_size_t *size) {
    
}

mapd_err_t FileMgr::getChunkActualSize(const ChunkKey &key, mapd_size_t *size) const {
    return MAPD_FAILURE;
}

mapd_err_t FileMgr::getBlock(const BlockAddr &blk, void *buf) const {
    
}

mapd_err_t FileMgr::getBlock(const int fileId, mapd_size_t blockAddr, void *buf) const {
    
}

mapd_err_t FileMgr::createBlock(const int fileId, mapd_size_t *blockAddr) {
    
}

mapd_err_t FileMgr::deleteBlock(const int fileId, mapd_size_t *blockAddr) {
    
}

mapd_err_t FileMgr::deleteBlock(const int fileId, const BlockAddr &index) {
    
}

void FileMgr::print() {
    
}

*/





