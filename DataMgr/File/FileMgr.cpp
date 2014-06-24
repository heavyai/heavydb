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

void FileMgr::print() {
	printf("FileMgr (%p)\n", this);
	printf("\tbasePath = \"%s\"\n", basePath_.c_str());
	printf("\tfile count = %lu\n", files_.size());
}
