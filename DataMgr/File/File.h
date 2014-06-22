/**
 * @file    File.h
 * @author  Steven Stewart <steve@map-d.com>
 * @brief   This file contains the class specification for File.
 *
 */
#ifndef _FILE_H
#define _FILE_H

#include <iostream>
#include <string>
#include "../../Shared/errors.h"
#include "../../Shared/types.h"

/**
 * @class  File
 * @author Steven Stewart <steve@map-d.com>
 * @brief  A class that implements basic file I/O.
 *
 * A file object represents data stored on physical disk. It's interface
 * facilitates reading and writing data to/from specific files. The client
 * can interact with the file at the byte or logical block level. The size
 * of a logical block is passed to the File via its constructor.
 *
 * @see Forbid copying idiom
 */
class File {
    
public:
    /**
     * A constructor that instantiates a File with a logical block size.
     * @see Explicit constructor idiom
     */ 
    explicit File(size_t blockSize);
    
    /**
     * The destructor must release the handle for the file (FILE *f_).
     */
    ~File();
    
    /**
     * Given the file name, this method opens a file handle. If the file
     * doesn't exist, then it is created. If the file exists and the
     * boolean "create" is false, then MAPD_ERR_FILE_OPEN is returned;
     * otherwise, a new empty file is created. Upon success, MAPD_SUCCESS
     * is returned.
     *
     * @param fname The file name to be opened.
     * @param create Indicates whether or not to create a new, empty file
     * @return MAPD_ERR_FILE_OPEN or MAPD_SUCCESS
     */
    mapd_err_t open(const std::string &fname, bool create = false);
    
    /**
     * @brief Closes the file.
     * The file is closed so that additional file I/O cannot be performed.
     *
     * @return MAPD_ERR_FILE_CLOSE or MAPD_SUCCESS
     */
    mapd_err_t close();
    
    /**
     * @brief Reads from the file to a buffer.
     * This method reads "n" bytes into the buffer "buf" starting at position
     * "pos" of the file.
     *
     * @param pos The starting position in the file from where to read.
     * @param n The number of bytes to read from the file.
     * @param buf A pointer to a memory buffer.
     * @return MAPD_ERR_FILE_READ or MAPD_SUCCESS
     */
    mapd_err_t read(_mapd_size_t pos, _mapd_size_t n, void *buf) const;
    
    /**
     * @brief Writes to the file from a buffer.
     * This method writes "n" bytes from the buffer "buf" to position
     * "pos" of the file.
     *
     * @param pos The starting position in the file for writing.
     * @param n The number of bytes to write to the file.
     * @param buf A pointer to a memory buffer to be written to the file.
     * @return MAPD_ERR_FILE_WRITE or MAPD_SUCCESS
     */
    mapd_err_t write(_mapd_size_t pos, _mapd_size_t n, void *buf);
    
    /**
     * @brief Appends data to a file from a buffer.
     * This method appends "n" bytes from the buffer "buf" to the end
     * of the file.
     *
     * @param n The number of bytes to append to the file.
     * @param buf A pointer to a memory buffer to be appended to the file.
     * @return MAPD_ERR_FILE_WRITE or MAPD_SUCCESS
     */
    mapd_err_t append(_mapd_size_t n, void *buf);
    
    /**
     * @brief Reads a specific block from the file to a buffer.
     * This method reads the contents a logical block into the buffer. The
     * block number is specified by blockNum.
     *
     * @param blockNum The logical block number in the file to be read from.
     * @param buf A pointer to a memory buffer where the block will be written.
     * @return MAPD_ERR_FILE_READ or MAPD_SUCCESS
     */
    mapd_err_t readBlock(_mapd_size_t blockNum, void *buf) const;
    
    /**
     * @brief Writes to the logical block in the file from a buffer.
     * This method writes the buffer "buf" to a logical block in the
     * file given by blockNum.
     *
     * @param blockNum The logical block being written to.
     * @param buf A pointer to a memory buffer that will be written to the file block.
     * @return MAPD_ERR_FILE_WRITE or MAPD_SUCCESS
     */
    mapd_err_t writeBlock(_mapd_size_t blockNum, void *buf);
    
    // Accessor(s) and Mutator(s)
    inline bool isOpen() const { return (f_ != NULL); }     /**< Returns true if the file exists. */
    inline size_t blockSize() const { return blockSize_; }  /**< Returns the logical block size. */
    inline size_t fileSize() const { return fileSize_; }    /**< Returns the file size in number of bytes. */
    inline void blockSize(size_t v) { blockSize_ = v; }     /**< Sets the logical block size to a new value. */
    
private:
    FILE *f_;               /**< a pointer to a file handle */
    std::string fileName_;  /**< the name of the file on physical disk */
    size_t blockSize_;      /**< the logical block size for the file */
    size_t fileSize_;       /**< the size of the file in bytes */
	
    /**
     * The copy constructor is made private to prevent attempts to copy a File object.
     * The rationale is that we don't want two different hooks to the file handle
     * at the same time.
     */
    File(const File&) {}
	
    /**
     * The assignment constructor is made private to prevent attempts to copy a File object.
     * The rationale is that we don't want two different hooks to the file handle
     * at the same time.
     */
    File& operator=(const File&);
	
}; // class File

#endif // _FILE_H