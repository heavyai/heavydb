/*
 * Copyright 2022 HEAVY.AI, Inc.
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
 * @brief   A selection of helper methods for File I/O.
 *
 */

#pragma once

#define DATA_FILE_EXT ".data"

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>

#include "Shared/types.h"

namespace File_Namespace {

constexpr auto kLegacyDataFileExtension{".mapd"};

std::string get_data_file_path(const std::string& base_path,
                               int file_id,
                               size_t page_size);

std::string get_legacy_data_file_path(const std::string& new_data_file_path);

std::pair<FILE*, std::string> create(const std::string& basePath,
                                     const int fileId,
                                     const size_t pageSize,
                                     const size_t npages);

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
 * @param size The number of bytes to be read.
 * @param buf The destination buffer to where data is being read from the file.
 * @param file_path Path of file to read from.
 * @return size_t The number of bytes read.
 */
size_t read(FILE* f,
            const size_t offset,
            const size_t size,
            int8_t* buf,
            const std::string& file_path);

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
size_t write(FILE* f, const size_t offset, const size_t size, const int8_t* buf);

/**
 * @brief Appends the specified number of bytes to the end of the file f from buf.
 *
 * @param f Pointer to the FILE.
 * @param n The number of bytes to append to the file.
 * @param buf The source buffer containing the data to be appended.
 * @param err If not NULL, will hold an error code should an error occur.
 * @return size_t The number of bytes written.
 */
size_t append(FILE* f, const size_t size, const int8_t* buf);

/**
 * @brief Reads the specified page from the file f into buf.
 *
 * @param f Pointer to the FILE.
 * @param pageSize The logical page size of the file.
 * @param pageNum The page number from where data is being read.
 * @param buf The destination buffer to where data is being written.
 * @param file_path Path of file to read from.
 * @return size_t The number of bytes read (should be equal to pageSize).
 */
size_t readPage(FILE* f,
                const size_t pageSize,
                const size_t pageNum,
                int8_t* buf,
                const std::string& file_path);

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

/**
 * @brief Renames a directory to DELETE_ME_<EPOCH>_<oldname>.
 * @param directoryName name of directory
 */
void renameForDelete(const std::string directoryName);

}  // namespace File_Namespace
