/**
 * @file		FileBuffer.h
 * @author		Steven Stewart <steve@map-d.com>
 * @author		Todd Mostak <todd@map-d.com>
 */

#ifndef DATAMGR_MEMORY_FILE_FILEBUFFER_H
#define DATAMGR_MEMORY_FILE_FILEBUFFER_H

#include "../AbstractBuffer.h"
#include "Page.h"

#include <iostream>
#include <stdexcept>

using namespace Memory_Namespace;

namespace File_Namespace {

    class FileMgr; // forward declaration

    /**
     * @class   FileBuffer
     * @brief   Represents/provides access to contiguous data stored in the file system.
     *
     * The FileBuffer consists of logical pages, which can map to any identically-sized
     * page in any file of the underlying file system. A page's metadata (file and page
     * number) are stored in MultiPage objects, and each MultiPage includes page
     * metadata for multiple versions of the same page.
     *
     * Note that a "Chunk" is brought into a FileBuffer by the FileMgr.
     *
     * Note(s): Forbid Copying Idiom 4.1
     */
    class FileBuffer : public AbstractBuffer {
        friend class FileMgr;
        
        public:
            
            /**
             * @brief Constructs a FileBuffer object.
             */
            FileBuffer(FileMgr *fm, const mapd_size_t pageSize, const ChunkKey &chunkKey, const mapd_size_t initialSize = 0);

            FileBuffer(FileMgr *fm, const mapd_size_t pageSize, const ChunkKey &chunkKey, const SQLTypes sqlType, const EncodingType encodingType, const EncodedDataType encodedDataType, const mapd_size_t initialSize = 0);

            FileBuffer(FileMgr *fm, const mapd_size_t pageSize, const ChunkKey &chunkKey, const std::vector<HeaderInfo>::const_iterator &headerStartIt, const std::vector<HeaderInfo>::const_iterator &headerEndIt);
            
            /// Destructor
            virtual ~FileBuffer();

            Page addNewMultiPage(const int epoch);

            void reserve(const mapd_size_t numBytes);

            void freePages();
            
            virtual void read(mapd_addr_t const dst, const mapd_size_t numBytes = 0, const BufferType dstBufferType = CPU_BUFFER, const mapd_size_t offset = 0);

            /**
             * @brief Writes the contents of source (src) into new versions of the affected logical pages.
             *
             * This method will write the contents of source (src) into new version of the affected
             * logical pages. New pages are only appended if the value of epoch (in FileMgr)
             *
             */
            virtual void write(mapd_addr_t src,  const mapd_size_t numBytes, const BufferType srcBufferType = CPU_BUFFER, const mapd_size_t offset = 0);

            virtual void append(mapd_addr_t src, const mapd_size_t numBytes, const BufferType srcBufferType = CPU_BUFFER);
            void copyPage(Page &srcPage, Page &destPage, const mapd_size_t numBytes, const mapd_size_t offset = 0);
            virtual inline Memory_Namespace::BufferType getType() const {return FILE_BUFFER;}

            /// Not implemented for FileMgr -- throws a runtime_error
            virtual mapd_byte_t* getMemoryPtr() {
                throw std::runtime_error("Operation not supported.");
            }

            /// Returns the number of pages in the FileBuffer.
            inline virtual mapd_size_t pageCount() const {
                return multiPages_.size();
            }
            
            /// Returns the size in bytes of each page in the FileBuffer.
            inline virtual mapd_size_t pageSize() const {
                return pageSize_;
            }

            inline virtual mapd_size_t size() const {
                return size_; 
            }
            
            /// Returns the total number of bytes allocated for the FileBuffer.
            inline virtual mapd_size_t reservedSize() const {
                return multiPages_.size() * pageSize_;
            }
            
            /// Returns the total number of used bytes in the FileBuffer.
            //inline virtual mapd_size_t used() const {
            
            /// Returns whether or not the FileBuffer has been modified since the last flush/checkpoint.
            virtual bool isDirty() const {
                return isDirty_;
            }

        private:
            //FileBuffer(const FileBuffer&);      // private copy constructor
            //FileBuffer& operator=(const FileBuffer&); // private overloaded assignment operator
            
            /// Write header writes header at top of page in format
            // headerSize(numBytes), ChunkKey, pageId, version epoch
            //void writeHeader(Page &page, const int pageId, const int epoch, const bool writeSize = false);
            void writeHeader(Page &page, const int pageId, const int epoch, const bool writeMetadata = false);
            void writeMetadata(const int epoch);
            void readMetadata(const Page &page);
            void calcHeaderBuffer();

            FileMgr *fm_; // a reference to FileMgr is needed for writing to new pages in available files
            static mapd_size_t headerBufferOffset_; 
            std::vector<MultiPage> multiPages_;
            ChunkKey chunkKey_;
            mapd_size_t pageSize_;
            mapd_size_t pageDataSize_;
            mapd_size_t reservedHeaderSize_; // lets make this a constant now for simplicity - 128 bytes
    };
    
} // File_Namespace

#endif // DATAMGR_MEMORY_FILE_FILEBUFFER_H


        
