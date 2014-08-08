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

    FILE* create(int fileId, mapd_size_t blockSize, mapd_size_t nblocks, mapd_err_t *err) {
        if (nblocks < 1 || blockSize < 1) {
            if (err) *err = MAPD_ERR_FILE_CREATE;
            return NULL;
        }

        FILE *f;
        std::string s = std::to_string(fileId) + std::string(MAPD_FILE_EXT);
        if ((f = fopen(s.c_str(), "w+b")) == NULL) {
            fprintf(stderr, "[%s:%d] Warning: unable to create file: fopen returned a NULL pointer.\n", __func__, __LINE__);
            return NULL;
        }

        fseek(f, (blockSize * nblocks)-1, SEEK_SET);
        fputc(EOF, f);
        fseek(f, 0, SEEK_SET); // rewind
        assert(fileSize(f) == (blockSize * nblocks));
        // fprintf(stdout, "[FileMgr] created file %d (blk_sz=%lu, nblocks=%lu, file_sz=%lu)\n", fileId, blockSize, nblocks, fileSize(f));
        
        if (err) {
            if ((fileSize(f) != blockSize * nblocks) || !f)
                *err = MAPD_ERR_FILE_CREATE;
            else
                *err = MAPD_SUCCESS;
        }
        return f;
    }

    FILE* open(int fileId, mapd_err_t *err) {
        FILE *f;
        std::string s = std::to_string(fileId) + std::string(MAPD_FILE_EXT);
        f = fopen(s.c_str(), "r+b"); // opens existing file for updates
        
        if (f != NULL && err)
            *err = MAPD_SUCCESS;
        else if (err)
            *err = MAPD_ERR_FILE_OPEN;
        
        return f;
    }
    
    mapd_err_t close(FILE *f) {
        assert(f);
        return (fclose(f) == 0) ? MAPD_SUCCESS : MAPD_ERR_FILE_CLOSE;
    }

    mapd_err_t removeFile(const std::string basePath, const std::string filename) {
        const std::string filePath = basePath + filename;
        if (remove(filePath.c_str()) != 0)
            return MAPD_FAILURE;
        return MAPD_SUCCESS;
    }
    
    size_t read(FILE *f, mapd_size_t offset, mapd_size_t n, mapd_addr_t buf, mapd_err_t *err) {
        assert(f);
        assert(buf);
        
        // read n bytes from the offset location in the file into the buffer
        fseek(f, offset, SEEK_SET);
        size_t bytesRead = fread(buf, sizeof(mapd_byte_t), n, f);
        if (bytesRead < 1) {
            fprintf(stderr, "[%s:%d] Error reading file contents into buffer.\n", __func__, __LINE__);
            if (err) *err = MAPD_ERR_FILE_READ;
        }
        else if (err)
            *err = MAPD_SUCCESS;
        return bytesRead;
    }
    
    size_t write(FILE *f, mapd_size_t offset, mapd_size_t n, mapd_addr_t buf, mapd_err_t *err) {
        assert(f);
        assert(buf);

        // write n bytes from the buffer to the offset location in the file
        fseek(f, offset, SEEK_SET);
        size_t bytesWritten = fwrite(buf, sizeof(mapd_byte_t), n, f);
        if (bytesWritten < 1) {
            fprintf(stderr, "[%s:%d] Error writing buffer contents to file.\n", __func__, __LINE__);
            if (err) *err = MAPD_ERR_FILE_WRITE;
        }
        else if (err)
            *err = MAPD_SUCCESS;
        return bytesWritten;
    }

    size_t append(FILE *f, mapd_size_t n, mapd_addr_t buf, mapd_err_t *err) {
        return write(f, fileSize(f), n, buf, err);
    }

    size_t readBlock(FILE *f, mapd_size_t blockSize, mapd_size_t blockNum, mapd_addr_t buf, mapd_err_t *err) {
        return read(f, blockNum * blockSize, blockSize, buf, err);
    }
    
    size_t writeBlock(FILE *f, mapd_size_t blockSize, mapd_size_t blockNum, mapd_addr_t buf, mapd_err_t *err) {
        return write(f, blockNum * blockSize, blockSize, buf, err);
    }
    
    size_t appendBlock(FILE *f, mapd_size_t blockSize, mapd_addr_t buf, mapd_err_t *err) {
        return write(f, fileSize(f), blockSize, buf, err);
    }
    
    /// @todo There may be an issue casting to size_t from long.
    size_t fileSize(FILE *f) {
        fseek(f, 0, SEEK_END);
        size_t size = (size_t)ftell(f);
        fseek(f, 0, SEEK_SET);
        return size;
    }

} // File_Namespace










