/**
 * @file		FileBuffer.h
 * @author		Steven Stewart <steve@map-d.com>
 */
#ifndef DATAMGR_MEMORY_FILEBUFFER_H
#define DATAMGR_MEMORY_FILEBUFFER_H

#include <iostream>
#include "../AbstractDatum.h"
#include "Block.h"

using namespace Memory_Namespace;

namespace File_Namespace {
    
    /**
     * @class   FileBuffer
     * @brief   Represents/provides access to contiguous data stored in the file system.
     *
     * The FileBuffer consists of logical blocks, which can map to any identically-sized
     * block in any file of the underlying file system. A block's metadata (file and block
     * number) are stored in MultiBlock objects, and each MultiBlock includes Block
     * metadata for multiple versions of the same block.
     *
     * Note(s): Forbid Copying Idiom 4.1
     */
    class FileBuffer : public AbstractDatum {
        friend class FileMgr;
        
    public:
        
        /**
         * @brief Constructs a FileBuffer object.
         */
        FileBuffer();
        
        /// Destructor
        virtual ~FileBuffer();
        
        virtual void read(mapd_addr_t const dst, const mapd_size_t offset, const mapd_size_t nbytes = 0);
        
        virtual void write(mapd_addr_t src, const mapd_size_t offset, const mapd_size_t nbytes);
        
        virtual void append(mapd_addr_t src, const mapd_size_t nbytes);
        
        /// Not implemented for FileMgr -- throws a runtime_error
        virtual const mapd_byte_t* getMemoryPtr() const {
            throw std::runtime_error("Operation not supported by FileMgr.");
        }
        
        /// Returns the number of pages in the FileBuffer.
        virtual mapd_size_t pageCount() const;
        
        /// Returns the size in bytes of each page in the FileBuffer.
        virtual mapd_size_t pageSize() const;
        
        /// Returns the total number of bytes allocated for the FileBuffer.
        virtual mapd_size_t size() const;
        
        /// Returns the total number of used bytes in the FileBuffer.
        virtual mapd_size_t used() const;
        
        /// Returns whether or not the FileBuffer has been modified since the last flush/checkpoint.
        virtual bool isDirty() const;
        
    private:
        FileBuffer(const FileBuffer&);      // private copy constructor
        FileBuffer& operator=(const FileBuffer&); // private overloaded assignment operator

        std::vector<MultiBlock> blocks_;
    };
    
} // File_Namespace

#endif // DATAMGR_MEMORY_FILEBUFFER_H