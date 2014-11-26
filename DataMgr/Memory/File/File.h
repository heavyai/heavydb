/**
 * @file    File.h
 * @author  Steven Stewart <steve@map-d.com>
 * @brief   A selection of helper methods for File I/O.
 *
 */
#ifndef DATAMGR_FILE_FILE_H
#define DATAMGR_FILE_FILE_H

#include <iostream>
#include <string>
#include "../../../Shared/errors.h"
#include "../../../Shared/types.h"

namespace File_Namespace {
    
    FILE* create(const int fileId, const mapd_size_t pageSize, const mapd_size_t npages);
    
    /**
     * @brief Opens/creates the file with the given id; returns NULL on error.
     *
     * @param fileId The id of the file to open.
     * @return FILE* A pointer to a FILE pointer, or NULL on error.
     */
    FILE* open(int fileId);
    
    /**
     * @brief Closes the file pointed to by the FILE pointer.
     *
     * @param f Pointer to the FILE.
     */
    void close(FILE *f);
    
    /**
     * @brief Deletes the file pointed to by the FILE pointer.
     *
     * @param basePath The base path (directory) of the file.
     * @param f Pointer to the FILE.
     * @return mapd_err_t Returns an error code when unable to close the file properly.
     */
    mapd_err_t removeFile(const std::string basePath, const std::string filename);
    
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
    size_t read(FILE *f, const mapd_size_t offset, const mapd_size_t size, mapd_addr_t buf);
    
    /**
     * @brief Writes the specified number of bytes to the offset position in file f from buf.
     *
     * @param f Pointer to the FILE.
     * @param offset The location within the file where data is being written.
     * @param size The number of bytes to write to the file.
     * @param buf The source buffer containing the data to be written.
     * @param err If not NULL, will hold an error code should an error occur.
     * @return size_t The number of bytes written.
     */
    size_t write(FILE *f, const mapd_size_t offset, const mapd_size_t size, mapd_addr_t buf);
    
    /**
     * @brief Appends the specified number of bytes to the end of the file f from buf.
     *
     * @param f Pointer to the FILE.
     * @param n The number of bytes to append to the file.
     * @param buf The source buffer containing the data to be appended.
     * @param err If not NULL, will hold an error code should an error occur.
     * @return size_t The number of bytes written.
     */
    size_t append(FILE *f, const mapd_size_t size, mapd_addr_t buf);
    
    /**
     * @brief Reads the specified page from the file f into buf.
     *
     * @param f Pointer to the FILE.
     * @param pageSize The logical page size of the file.
     * @param pageNum The page number from where data is being read.
     * @param buf The destination buffer to where data is being written.
     * @param err If not NULL, will hold an error code should an error occur.
     * @return size_t The number of bytes read (should be equal to pageSize).
     */
    size_t readPage(FILE *f, const mapd_size_t pageSize, const mapd_size_t pageNum, mapd_addr_t buf);
    
    /**
     * @brief Writes a page from buf to the file.
     *
     * @param f Pointer to the FILE.
     * @param pageSize The logical page size of the file.
     * @param pageNum The page number to where data is being written.
     * @param buf The source buffer from where data is being read.
     * @param err If not NULL, will hold an error code should an error occur.
     * @return size_t The number of bytes written (should be equal to pageSize).
     */
    size_t writePage(FILE *f, const mapd_size_t pageSize, const mapd_size_t pageNum, mapd_addr_t buf);
    
    /**
     * @brief Appends a page from buf to the file.
     *
     * @param f Pointer to the FILE.
     * @param pageSize The logical page size of the file.
     * @param buf The source buffer from where data is being read.
     * @param err If not NULL, will hold an error code should an error occur.
     * @return size_t The number of bytes appended (should be equal to pageSize).
     */
    size_t appendPage(FILE *f, const mapd_size_t pageSize, mapd_addr_t buf);
    
    /**
     * @brief Returns the size of the specified file.
     * @param f A pointer to the file.
     * @return size_t The number of bytes of the file.
     */
    size_t fileSize(FILE *f);
    
} // namespace File_Namespace

#endif // DATAMGR_FILE_FILE_H
