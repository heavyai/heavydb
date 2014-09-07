/**
 * @file        FileBuffer.cpp
 * @author      Steven Stewart <steve@map-d.com>
 */
#include "FileBuffer.h"

namespace File_Namespace {
    
    FileBuffer::FileBuffer() {

    }
    
    FileBuffer::FileBuffer() {

    }
    
    void FileBuffer::read(mapd_addr_t const dst, const mapd_size_t offset, const mapd_size_t nbytes = 0) {
        
    }
    
    /// this method returns 0 if it cannot write the full n bytes
    size_t FileBuffer::write(mapd_size_t offset, mapd_size_t n, mapd_addr_t src) {

    }
    
    /// this method returns 0 if it cannot append the full n bytes
    size_t FileBuffer::append(mapd_size_t n, mapd_addr_t src) {

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
