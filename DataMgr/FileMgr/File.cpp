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
#include <stdexcept>
#include <unistd.h>
#include "File.h"

namespace File_Namespace {
    
    FILE* create(const std::string &basePath, const int fileId, const size_t pageSize, const size_t numPages) {
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

    FILE* create(const std::string &fullPath, const size_t requestedFileSize) {
        if (requestedFileSize <= 0) {
            throw std::invalid_argument("Created file size must be > 0");
        }
        FILE *f;
        if ((f = fopen(fullPath.c_str(), "w+b")) == NULL)
            throw std::runtime_error ("Unable to create file");

        fseek(f, requestedFileSize-1, SEEK_SET);
        fputc(EOF, f);
        fseek(f, 0, SEEK_SET); // rewind
        assert(fileSize(f) == requestedFileSize);
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
        if (fflush(f) != 0)
          throw std::runtime_error("Unable to flush file.");
        if (fclose(f) != 0)
            throw std::runtime_error("Unable to close file.");
        
    }
    
    bool removeFile(const std::string basePath, const std::string filename) {
        const std::string filePath = basePath + filename;
        return remove(filePath.c_str()) == 0;
    }
    
    size_t read(FILE *f, const size_t offset, const size_t size, int8_t * buf) {
        //assert(f);
        //assert(buf);
        //assert(size > 0);
        
        // read "size" bytes from the offset location in the file into the buffer
        fseek(f, offset, SEEK_SET);
        size_t bytesRead = fread(buf, sizeof(int8_t), size, f);
        //size_t bytesRead = fread(buf, sizeof(int8_t)*size,1, f) * size;
        if (bytesRead < 1)
            throw std::runtime_error("Error reading file contents into buffer.");
        return bytesRead;
    }
    
    size_t write(FILE *f, const size_t offset, const size_t size, int8_t * buf) {
        //assert(f);
        //assert(buf);
        // write size bytes from the buffer to the offset location in the file
        fseek(f, offset, SEEK_SET);
        size_t bytesWritten = fwrite(buf, sizeof(int8_t), size, f);
        //size_t bytesWritten = fwrite(buf, sizeof(int8_t)*size,1, f) * size;
        if (bytesWritten < 1)
            throw std::runtime_error("Error writing buffer to file.");
        return bytesWritten;
    }
    
    size_t append(FILE *f, const size_t size, int8_t * buf) {
        return write(f, fileSize(f), size, buf);
    }
    
    size_t readPage(FILE *f, const size_t pageSize, const size_t pageNum, int8_t * buf) {
        return read(f, pageNum * pageSize, pageSize, buf);
    }

    size_t readPartialPage(FILE *f, const size_t pageSize, const size_t offset, const size_t readSize, const size_t pageNum, int8_t * buf) {
        return read(f, pageNum * pageSize + offset, readSize, buf);
    }
    
    size_t writePage(FILE *f, const size_t pageSize, const size_t pageNum, int8_t * buf) {
        return write(f, pageNum * pageSize, pageSize, buf);
    }

    size_t writePartialPage(FILE *f, const size_t pageSize, const size_t offset, const size_t writeSize, const size_t pageNum, int8_t * buf) {
        return write(f, pageNum * pageSize + offset, writeSize, buf);
    }
    
    size_t appendPage(FILE *f, const size_t pageSize, int8_t * buf) {
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

