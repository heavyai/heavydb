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
#include "FileMgr.h"


FileMgr::FileMgr(const std::string &basePath) {
	basePath_ = basePath;
}

FileMgr::~FileMgr() {
    
}

/**
 * @brief Finds the chunk using the key, and returns the reference in c.
 */
mapd_err_t FileMgr::getChunkRef(const ChunkKey &key, Chunk &c) const {
    ChunkKeyToChunkMap::iterator iter = chunkIndex_.find(key);
    if (iter != chunkIndex_.end()) {
        // found
        c = iter->second;
        return MAPD_SUCCESS;
    }
    else {
        // not found
        return MAPD_ERR_CHUNK_NOT_FOUND; // chunk doesn't exist
    }
}

mapd_err_t FileMgr::getChunkCopy(const ChunkKey &key, void *buf) const {
    
}

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

mapd_err_t FileMgr::addFile(const std::string &fileName, const mapd_size_t blockSize, const mapd_size_t numBlocks, int *fileId) {
    
}

mapd_err_t FileMgr::deleteFile(const int fileId) {
    
}

void FileMgr::print() {
    
}







