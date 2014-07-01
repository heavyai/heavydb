#include <iostream>
#include <cassert>
#include "File.h"

#define MAPD_FILE_EXT ".mapd"

namespace File {

    FILE* open(int fileId, bool create, mapd_err_t *err) {
        FILE *f;
        std::string s = std::to_string(fileId) + std::string(MAPD_FILE_EXT);

        if (create)
            f = fopen(s.c_str(), "w+b"); // creates new file for updates
        else
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
    
    size_t read(FILE *f, mapd_size_t offset, mapd_size_t n, void *buf, mapd_err_t *err) {
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
    
    size_t write(FILE *f, mapd_size_t offset, mapd_size_t n, void *buf, mapd_err_t *err) {
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

    size_t append(FILE *f, mapd_size_t n, void *buf, mapd_err_t *err) {
        return write(f, fileSize(f), n, buf, err);
    }

    size_t readBlock(FILE *f, mapd_size_t blockSize, mapd_size_t blockNum, void *buf, mapd_err_t *err) {
        return read(f, blockNum * blockSize, blockSize, buf, err);
    }
    
    size_t writeBlock(FILE *f, mapd_size_t blockSize, mapd_size_t blockNum, void *buf, mapd_err_t *err) {
        return write(f, blockNum * blockSize, blockSize, buf, err);
    }
    
    size_t appendBlock(FILE *f, mapd_size_t blockSize, void *buf, mapd_err_t *err) {
        return write(f, fileSize(f), blockSize, buf, err);
    }
    
    /// @todo There may be an issue casting to size_t from long.
    size_t fileSize(FILE *f) {
        fseek(f, 0, SEEK_END);
        size_t size = (size_t)ftell(f);
        fseek(f, 0, SEEK_SET);
        return size;
    }

}










