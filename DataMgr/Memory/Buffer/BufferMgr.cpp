//
//  BufferMgr.cpp
//  mapd2
//
//  Created by Steven Stewart on 8/27/14.
//  Copyright (c) 2014 Map-D Technologies, Inc. All rights reserved.
//

#include "BufferMgr.h"

namespace Buffer_Namespace {

    BufferMgr::BufferMgr(mapd_size_t pageSize, mapd_size_t numPages) {
        memSize_ = pageSize * numPages;
        assert(memSize_ > 0);
        mem_ = (mapd_addr_t) new mapd_byte_t[memSize_];
        freeMem_.insert(std::pair<mapd_size_t, mapd_addr_t>(memSize_, mem_));
    }

    BufferMgr::~BufferMgr() {
        delete[] mem_;
    }
    
        
}