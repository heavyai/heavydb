/**
 * @file    File.cpp
 * @author  Steven Stewart <steve@map-d.com>
 * @brief   Implementation of helper methods for File I/O.
 *
 */
#include <iostream>
#include <cassert>
#include <cstdio>
#include <string>
#include <unistd.h>
#include "File.h"

#define MAPD_FILE_EXT ".mapd"

namespace File_Namespace {
    
    FILE* create(const int fileId, const mapd_size_t blockSize, const mapd_size_t nblocks) {
        if (nblocks < 1 || blockSize < 1)
            throw std::invalid_argument("Number of blocks and block size must be positive integers.");
        
        FILE *f;
        std::string s = std::to_string(fileId) + std::string(MAPD_FILE_EXT);
        if ((f = fopen(s.c_str(), "w+b")) == NULL) {
            fprintf(stderr, "[%s:%d] Warning: unable to create file: fopen returned a NULL pointer.\n", __func__, __LINE__);
            return nullptr;
        }
        
        fseek(f, (blockSize * nblocks)-1, SEEK_SET);
        fputc(EOF, f);
        fseek(f, 0, SEEK_SET); // rewind
        assert(fileSize(f) == (blockSize * nblocks));
        
        return f;
    }
    
    FILE* open(int fileId) {
        FILE *f;
        std::string s = std::to_string(fileId) + std::string(MAPD_FILE_EXT);
        f = fopen(s.c_str(), "r+b"); // opens existing file for updates
        if (f == nullptr)
            throw std::runtime_error("Unable to open file.");
        return f;
    }
    
    void close(FILE *f) {
        assert(f);
        fflush(f);
        if (fsync(fileno(f)) != 0)
            throw std::runtime_error("Unable to close file.");
        if (fclose(f) != 0)
            throw std::runtime_error("Unable to close file.");
        
    }
    
    mapd_err_t removeFile(const std::string basePath, const std::string filename) {
        const std::string filePath = basePath + filename;
        if (remove(filePath.c_str()) != 0)
            return MAPD_FAILURE;
        return MAPD_SUCCESS;
    }
    
    size_t read(FILE *f, const mapd_size_t offset, const mapd_size_t size, mapd_addr_t buf) {
        assert(f);
        assert(buf);
        assert(size > 0);
        
        // read "size" bytes from the offset location in the file into the buffer
        fseek(f, offset, SEEK_SET);
        size_t bytesRead = fread(buf, sizeof(mapd_byte_t), size, f);
        if (bytesRead < 1)
            throw std::runtime_error("Error reading file contents into buffer.");
        return bytesRead;
    }
    
    size_t write(FILE *f, const mapd_size_t offset, const mapd_size_t size, mapd_addr_t buf) {
        assert(f);
        assert(buf);
        
        // write size bytes from the buffer to the offset location in the file
        fseek(f, offset, SEEK_SET);
        size_t bytesWritten = fwrite(buf, sizeof(mapd_byte_t), size, f);
        if (bytesWritten < 1)
            throw std::runtime_error("Error writing buffer to file.");
        return bytesWritten;
    }
    
    size_t append(FILE *f, const mapd_size_t size, mapd_addr_t buf) {
        return write(f, fileSize(f), size, buf);
    }
    
    size_t readBlock(FILE *f, const mapd_size_t blockSize, const mapd_size_t blockNum, mapd_addr_t buf) {
        return read(f, blockNum * blockSize, blockSize, buf);
    }
    
    size_t writeBlock(FILE *f, const mapd_size_t blockSize, const mapd_size_t blockNum, mapd_addr_t buf) {
        return write(f, blockNum * blockSize, blockSize, buf);
    }
    
    size_t appendBlock(FILE *f, const mapd_size_t blockSize, mapd_addr_t buf) {
        return write(f, fileSize(f), blockSize, buf);
    }
    
    /// @todo There may be an issue casting to size_t from long.
    size_t fileSize(FILE *f) {
        fseek(f, 0, SEEK_END);
        size_t size = (size_t)ftell(f);
        fseek(f, 0, SEEK_SET);
        return size;
    }
    
} // File_Namespace

