/*
 * Copyright 2017 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file    File.h
 * @author  Steven Stewart <steve@map-d.com>
 * @brief   A selection of helper methods for File I/O.
 *
 */
#ifndef DATAMGR_FILE_FILE_H
#define DATAMGR_FILE_FILE_H

#define MAPD_FILE_EXT ".mapd"
#define MAX_FILE_N_PAGES 256
#define MAX_FILE_N_METADATA_PAGES 4096

#include <iostream>
#include <string>
#include "../../Shared/types.h"

namespace File_Namespace {

FILE* create(const std::string& basePath, const int fileId, const size_t pageSize, const size_t npages);

FILE* create(const std::string& fullPath, const size_t requestedFileSize);

/**
 * @brief Opens/creates the file with the given id; returns NULL on error.
 *
 * @param fileId The id of the file to open.
 * @return FILE* A pointer to a FILE pointer, or NULL on error.
 */
FILE* open(int fileId);

FILE* open(const std::string& path);

/**
 * @brief Closes the file pointed to by the FILE pointer.
 *
 * @param f Pointer to the FILE.
 */
void close(FILE* f);

/**
 * @brief Deletes the file pointed to by the FILE pointer.
 *
 * @param basePath The base path (directory) of the file.
 * @param f Pointer to the FILE.
 * @return mapd_err_t Returns an error code when unable to close the file properly.
 */
bool removeFile(const std::string basePath, const std::string filename);

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
size_t read(FILE* f, const size_t offset, const size_t size, int8_t* buf);

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
size_t write(FILE* f, const size_t offset, const size_t size, int8_t* buf);

/**
 * @brief Appends the specified number of bytes to the end of the file f from buf.
 *
 * @param f Pointer to the FILE.
 * @param n The number of bytes to append to the file.
 * @param buf The source buffer containing the data to be appended.
 * @param err If not NULL, will hold an error code should an error occur.
 * @return size_t The number of bytes written.
 */
size_t append(FILE* f, const size_t size, int8_t* buf);

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
size_t readPage(FILE* f, const size_t pageSize, const size_t pageNum, int8_t* buf);

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
size_t writePage(FILE* f, const size_t pageSize, const size_t pageNum, int8_t* buf);

/**
 * @brief Appends a page from buf to the file.
 *
 * @param f Pointer to the FILE.
 * @param pageSize The logical page size of the file.
 * @param buf The source buffer from where data is being read.
 * @param err If not NULL, will hold an error code should an error occur.
 * @return size_t The number of bytes appended (should be equal to pageSize).
 */
size_t appendPage(FILE* f, const size_t pageSize, int8_t* buf);

/**
 * @brief Returns the size of the specified file.
 * @param f A pointer to the file.
 * @return size_t The number of bytes of the file.
 */
size_t fileSize(FILE* f);

}  // namespace File_Namespace

#endif  // DATAMGR_FILE_FILE_H
