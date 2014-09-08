/**
 * @file        FileBuffer.cpp
 * @author      Steven Stewart <steve@map-d.com>
 */
#include "FileBuffer.h"
#include "File.h"
#include <map>

namespace File_Namespace {
    
    FileBuffer::FileBuffer() {
        // NOP
    }
    
    FileBuffer::~FileBuffer() {
        // NOP
    }

    void FileBuffer::read(mapd_addr_t const dst, const mapd_size_t offset, const mapd_size_t nbytes) {
        size_t nblocks = blocks_.size();
        mapd_addr_t cur = dst + offset;
        
        std::map<int, FILE*> openFiles;
        
        // Traverse the blocks
        for (size_t i = 0; i < nblocks; ++i) {
            // Obtain the most recent version of the current block
            Block *b = blocks_[i].current();
            size_t blockSize = blocks_[i].blockSize;
            printf("fileId=%d blockNum=%lu blockSize=%u\n", b->fileId, b->blockNum, blockSize);
            
            // Open the file
            FILE *f = File_Namespace::open(b->fileId);
            assert(f);
            
            // Read the block into the destination (dst) buffer at its
            // current (cur) location
            size_t bytesRead = File_Namespace::readBlock(f, blockSize, b->blockNum, cur);
            assert(bytesRead == blockSize);
            
            cur += blockSize;
        }
        
        // Close any open files
        for (auto fileIt = openFiles.begin(); fileIt != openFiles.end(); ++fileIt)
            close(fileIt->second);
    }
    
    /// this method returns 0 if it cannot write the full n bytes
    void FileBuffer::write(mapd_addr_t src, const mapd_size_t offset, const mapd_size_t nbytes) {
        
    }

    /// this method returns 0 if it cannot append the full n bytes
    void FileBuffer::append(mapd_addr_t src, const mapd_size_t nbytes) {
        
    }
    
    /// this method returns 0 if it cannot copy the full n bytes
    size_t FileBuffer::copy(mapd_size_t offset, mapd_size_t n, mapd_addr_t dest) {

    }
    
    std::vector<bool> FileBuffer::getDirty() {

    }
    
    void FileBuffer::print() {

    }
    
    void FileBuffer::print(mapd_data_t type) {

    }
    
} // File_Namespace
