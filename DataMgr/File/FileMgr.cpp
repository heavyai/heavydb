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

mapd_err_t addFile(const std::string &fileName, const mapd_size_t blockSize, const mapd_size_t numBlocks, int *fileId) {
    *fileId = files_.size();
    
}
