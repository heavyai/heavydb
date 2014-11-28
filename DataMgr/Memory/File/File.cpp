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


namespace File_Namespace {
    
    FILE* create(const std::string &basePath, const int fileId, const mapd_size_t pageSize, const mapd_size_t numPages) {
        if (numPages < 1 || pageSize < 1)
            throw std::invalid_argument("Number of pages and page size must be positive integers.");
        
        FILE *f;
        std::string path (basePath + std::to_string(fileId) + "." + std::to_string(pageSize) +  std::string(MAPD_FILE_EXT)); // MAPD_FILE_EXT has preceding "."
        if ((f = fopen(path.c_str(), "w+b")) == NULL)
            throw std::runtime_error("Unable to create file");
        
        fseek(f, (pageSize * numPages)-1, SEEK_SET);
        fputc(EOF, f);
        fseek(f, 0, SEEK_SET); // rewind
        assert(fileSize(f) == (pageSize * numPages));
        
        return f;
    }
    
    FILE* open(int fileId) {
        FILE *f;
        std::string s (std::to_string(fileId) + std::string(MAPD_FILE_EXT));
        f = fopen(s.c_str(), "r+b"); // opens existing file for updates
        if (f == nullptr)
            throw std::runtime_error("Unable to open file.");
        return f;
    }

    FILE* open(const std::string &path) {
        FILE *f;
        f = fopen(path.c_str(), "r+b"); // opens existing file for updates
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
    
    size_t readPage(FILE *f, const mapd_size_t pageSize, const mapd_size_t pageNum, mapd_addr_t buf) {
        return read(f, pageNum * pageSize, pageSize, buf);
    }

    size_t readPartialPage(FILE *f, const mapd_size_t pageSize, const mapd_size_t offset, const mapd_size_t readSize, const mapd_size_t pageNum, mapd_addr_t buf) {
        return read(f, pageNum * pageSize + offset, readSize, buf);
    }
    
    size_t writePage(FILE *f, const mapd_size_t pageSize, const mapd_size_t pageNum, mapd_addr_t buf) {
        return write(f, pageNum * pageSize, pageSize, buf);
    }

    size_t writePartialPage(FILE *f, const mapd_size_t pageSize, const mapd_size_t offset, const mapd_size_t writeSize, const mapd_size_t pageNum, mapd_addr_t buf) {
        return write(f, pageNum * pageSize + offset, writeSize, buf);
    }
    
    size_t appendPage(FILE *f, const mapd_size_t pageSize, mapd_addr_t buf) {
        return write(f, fileSize(f), pageSize, buf);
    }
    
    /// @todo There may be an issue casting to size_t from long.
    size_t fileSize(FILE *f) {
        fseek(f, 0, SEEK_END);
        size_t size = (size_t)ftell(f);
        fseek(f, 0, SEEK_SET);
        return size;
    }
    
} // File_Namespace

