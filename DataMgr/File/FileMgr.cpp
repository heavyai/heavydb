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
    FileMap::iterator iter = files_.find(fileId);

    if (iter != files_.end()) { // delete FileInfo object
        // obtain FileInfo pointer
        FileInfo *fInfo = findFile(fileId, NULL);
        
        // close open file handle
        if (fInfo->f)
            File::close(fInfo->f);
        
        // free memory used by FileInfo object
        delete fInfo;
        fInfo = NULL;
        
        // remove FileMap entry
        files_.erase(iter);
        return MAPD_SUCCESS;
    }
    return MAPD_FAILURE;
}

FileInfo* FileMgr::findFile(const int fileId, mapd_err_t *err) {
    FileMap::iterator iter = files_.find(fileId);
    if (iter != files_.end()) {
        if (err) *err = MAPD_SUCCESS;
        return iter->second;
    }
    if (err) *err = MAPD_ERR_FILE_NOT_FOUND;
    return NULL;
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





