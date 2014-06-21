/**
 *  File:       File.h
 *  Author(s):  steve@map-d.com
 */
#ifndef _FILE_H
#define _FILE_H

#include <iostream>
#include <string>
#include "../../Shared/errors.h"
#include "../../Shared/types.h"

/**
 *  class:      File
 *  Author(s):  steve@map-d.com
 * 
 *  A file object represents data stored on physical disk. It's interface
 *  facilitates reading and writing data to/from specific files.
 * 
 */
class File {
    
public:
    File();

    // Operations
    mapd_err_t open(const std::string &fname, bool create = false);
    mapd_err_t close();
    mapd_err_t read(_mapd_size_t pos, _mapd_size_t n, void *buf) const;
    mapd_err_t write(_mapd_size_t pos, _mapd_size_t n, void *buf) const;
    mapd_err_t append(_mapd_size_t n, void *buf) const;
    mapd_err_t readBlock(_mapd_size_t blockNum, void *buf) const;
    mapd_err_t writeBlock(_mapd_size_t blockNum, void *buf) const;
    mapd_err_t appendBlock();
    
    // Accessor(s) and Mutator(s)
    inline bool isOpen() const { return (f_ == NULL); }
    inline int blockSize() const { return blockSize_; }
    inline int fileSize() const { return fileSize_; }
    
    inline void blockSize(unsigned v) { blockSize_ = v; }
    inline void fileSize(unsigned v) { fileSize_ = v; }
    inline void fileName(const std::string &fname) { fileName_ = fname; }
    
private:
    // properties
    FILE *f_;
    std::string fileName_;
    unsigned blockSize_;
    unsigned fileSize_;
};

#endif // _FILE_H