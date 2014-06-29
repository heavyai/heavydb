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

mapd_err_t FileMgr::getChunkActualSize(const ChunkKey &key, mapd_size_t *size) const {
    return MAPD_FAILURE;
}