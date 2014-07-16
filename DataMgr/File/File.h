/**
 * @file    File.h
 * @author  Steven Stewart <steve@map-d.com>
 * @brief   A selection of helper methods for File I/O.
 *
 */
#ifndef _FILE_H
#define _FILE_H

#include <iostream>
#include <string>
#include "../../Shared/errors.h"
#include "../../Shared/types.h"

namespace File_Namespace {

    FILE* create(int fileId, mapd_size_t blockSize, mapd_size_t nblocks, mapd_err_t *err);

    /**
     * @brief Opens/creates the file with the given id; returns NULL on error.
     *
     * @param fileId The id of the file to open.
     * @param create A flag indicating whether or not to create a new file
     * @param err An error code, should an error occur.
     * @return FILE* A pointer to a FILE pointer, or NULL on error.
     */
    FILE* open(int fileId, mapd_err_t *err);
    
    /**
     * @brief Closes the file pointed to by the FILE pointer.
     *
     * @param f Pointer to the FILE.
     * @return mapd_err_t Returns an error code when unable to close the file properly.
     */
    mapd_err_t close(FILE *f);

    /**
     * @brief Deletes the file pointed to by the FILE pointer.
     *
     * @param basePath The base path (directory) of the file.
     * @param f Pointer to the FILE.
     * @return mapd_err_t Returns an error code when unable to close the file properly.
     */
    mapd_err_t erase(const std::string basePath, FILE *f);

    /**
     * @brief Reads the specified number of bytes from the offset position in file f into buf.
     *
     * @param f Pointer to the FILE.
     * @param offset The location within the file from which to read.
     * @param n The number of bytes to be read.
     * @param buf The destination buffer to where data is being read from the file.
     * @param err If not NULL, will hold an error code should an error occur.
     * @return size_t The number of bytes read.
     */
    size_t read(FILE *f, mapd_size_t offset, mapd_size_t n, mapd_byte_t *buf, mapd_err_t *err);
    
    /**
     * @brief Writes the specified number of bytes to the offset position in file f from buf.
     *
     * @param f Pointer to the FILE.
     * @param offset The location within the file where data is being written.
     * @param n The number of bytes to write to the file.
     * @param buf The source buffer containing the data to be written.
     * @param err If not NULL, will hold an error code should an error occur.
     * @return size_t The number of bytes written.
     */
    size_t write(FILE *f, mapd_size_t offset, mapd_size_t n, mapd_byte_t *buf, mapd_err_t *err);
    
   /**
    * @brief Appends the specified number of bytes to the end of the file f from buf.
    *
    * @param f Pointer to the FILE.
    * @param n The number of bytes to append to the file.
    * @param buf The source buffer containing the data to be appended.
    * @param err If not NULL, will hold an error code should an error occur.
    * @return size_t The number of bytes written.
    */
    size_t append(FILE *f, mapd_size_t n, mapd_byte_t *buf, mapd_err_t *err);
    
    /**
     * @brief Reads the specified block from the file f into buf.
     *
     * @param f Pointer to the FILE.
     * @param blockSize The logical block size of the file.
     * @param blockNum The block number from where data is being read.
     * @param buf The destination buffer to where data is being written.
     * @param err If not NULL, will hold an error code should an error occur.
     * @return size_t The number of bytes read (should be equal to blockSize).
     */
    size_t readBlock(FILE *f, mapd_size_t blockSize, mapd_size_t blockNum, mapd_byte_t *buf, mapd_err_t *err);
    
    /**
     * @brief Writes a block from buf to the file.
     *
     * @param f Pointer to the FILE.
     * @param blockSize The logical block size of the file.
     * @param blockNum The block number to where data is being written.
     * @param buf The source buffer from where data is being read.
     * @param err If not NULL, will hold an error code should an error occur.
     * @return size_t The number of bytes written (should be equal to blockSize).
     */
    size_t writeBlock(FILE *f, mapd_size_t blockSize, mapd_size_t blockNum, mapd_byte_t *buf, mapd_err_t *err);

    /**
     * @brief Appends a block from buf to the file.
     *
     * @param f Pointer to the FILE.
     * @param blockSize The logical block size of the file.
     * @param blockNum The block number to where data is being appended.
     * @param buf The source buffer from where data is being read.
     * @param err If not NULL, will hold an error code should an error occur.
     * @return size_t The number of bytes appended (should be equal to blockSize).
     */
    size_t appendBlock(FILE *f, mapd_size_t blockSize, mapd_byte_t *buf, mapd_err_t *err);

    /**
     * @brief Returns the size of the specified file.
     * @param f A pointer to the file.
     * @return size_t The number of bytes of the file.
     */
    size_t fileSize(FILE *f);

} // namespace File_Namespace

#endif // _FILE_H
