/**
 * @file	FileMgr.h
 * @author	Steven Stewart <steve@map-d.com>
 *
 * This file includes the class specification for the FILE manager (FileMgr), and related
 * data structures and types.
 */
#ifndef DATAMGR_MEMORY_FILE_FILEMGR_H
#define DATAMGR_MEMORY_FILE_FILEMGR_H

#include <iostream>
#include "../AbstractDatum.h"
#include "../AbstractDataMgr.h"

using namespace Memory_Namespace;

namespace File_Namespace {
    
    /**
     * @class   FileMgr
     * @brief
     */
    class FileMgr : public AbstractDataMgr { // implements
        
        FileMgr(std::string basePath = ".");
        
        virtual ~FileMgr();
        
        // Chunk API
        virtual void createChunk(const ChunkKey &key, mapd_size_t pageSize, mapd_size_t nbytes = 0, mapd_addr_t buf = nullptr);
        
        virtual void deleteChunk(const ChunkKey &key);
        virtual void releaseChunk(const ChunkKey &key);
        virtual void copyChunkToDatum(const ChunkKey &key, AbstractDatum *datum);
        
        // Datum API
        virtual void createDatum(mapd_size_t pageSize, mapd_size_t nbytes = 0);
        virtual void deleteDatum(int id);
        
    };
    
} // File_Namespace

#endif // DATAMGR_MEMORY_FILE_FILEMGR_H