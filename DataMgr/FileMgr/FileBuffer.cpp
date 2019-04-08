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
 * @file        FileBuffer.cpp
 * @author      Steven Stewart <steve@map-d.com>
 * @author      Todd Mostak <todd@map-d.com>
 */

#include "FileBuffer.h"
#include <glog/logging.h>
#include <future>
#include <map>
#include <thread>
#include "../../Shared/File.h"
#include "FileMgr.h"

#define METADATA_PAGE_SIZE 4096

using namespace std;

namespace File_Namespace {
size_t FileBuffer::headerBufferOffset_ = 32;

FileBuffer::FileBuffer(FileMgr* fm,
                       const size_t pageSize,
                       const ChunkKey& chunkKey,
                       const size_t initialSize)
    : AbstractBuffer(fm->getDeviceId())
    , fm_(fm)
    , metadataPages_(METADATA_PAGE_SIZE)
    , pageSize_(pageSize)
    , chunkKey_(chunkKey) {
  // Create a new FileBuffer
  CHECK(fm_);
  calcHeaderBuffer();
  pageDataSize_ = pageSize_ - reservedHeaderSize_;
  //@todo reintroduce initialSize - need to develop easy way of
  // differentiating these pre-allocated pages from "written-to" pages
  /*
  if (initalSize > 0) {
      // should expand to initialSize bytes
      size_t initialNumPages = (initalSize + pageSize_ -1) / pageSize_;
      int epoch = fm_->epoch();
      for (size_t pageNum = 0; pageNum < initialNumPages; ++pageNum) {
          Page page = addNewMultiPage(epoch);
          writeHeader(page,pageNum,epoch);
      }
  }
  */
}

FileBuffer::FileBuffer(FileMgr* fm,
                       const size_t pageSize,
                       const ChunkKey& chunkKey,
                       const SQLTypeInfo sqlType,
                       const size_t initialSize)
    : AbstractBuffer(fm->getDeviceId(), sqlType)
    , fm_(fm)
    , metadataPages_(METADATA_PAGE_SIZE)
    , pageSize_(pageSize)
    , chunkKey_(chunkKey) {
  CHECK(fm_);
  calcHeaderBuffer();
  pageDataSize_ = pageSize_ - reservedHeaderSize_;
}

FileBuffer::FileBuffer(FileMgr* fm,
                       /* const size_t pageSize,*/ const ChunkKey& chunkKey,
                       const std::vector<HeaderInfo>::const_iterator& headerStartIt,
                       const std::vector<HeaderInfo>::const_iterator& headerEndIt)
    : AbstractBuffer(fm->getDeviceId())
    , fm_(fm)
    , metadataPages_(METADATA_PAGE_SIZE)
    , pageSize_(0)
    , chunkKey_(chunkKey) {
  // We are being assigned an existing FileBuffer on disk

  CHECK(fm_);
  calcHeaderBuffer();
  // MultiPage multiPage(pageSize_); // why was this here?
  int lastPageId = -1;
  // Page lastMetadataPage;
  for (auto vecIt = headerStartIt; vecIt != headerEndIt; ++vecIt) {
    int curPageId = vecIt->pageId;

    // We only want to read last metadata page
    if (curPageId == -1) {  // stats page
      metadataPages_.epochs.push_back(vecIt->versionEpoch);
      metadataPages_.pageVersions.push_back(vecIt->page);
    } else {
      if (curPageId != lastPageId) {
        // protect from bad data on disk, and give diagnostics
        if (curPageId != lastPageId + 1) {
          LOG(FATAL) << "Failure reading DB file " << showChunk(chunkKey)
                     << " Current page " << curPageId << " last page " << lastPageId
                     << " epoch " << vecIt->versionEpoch;
        }
        if (lastPageId == -1) {
          // If we are on first real page
          CHECK(metadataPages_.pageVersions.back().fileId != -1);  // was initialized
          readMetadata(metadataPages_.pageVersions.back());
          pageDataSize_ = pageSize_ - reservedHeaderSize_;
        }
        MultiPage multiPage(pageSize_);
        multiPages_.push_back(multiPage);
        lastPageId = curPageId;
      }
      multiPages_.back().epochs.push_back(vecIt->versionEpoch);
      multiPages_.back().pageVersions.push_back(vecIt->page);
    }
    if (curPageId == -1) {  // meaning there was only a metadata page
      readMetadata(metadataPages_.pageVersions.back());
      pageDataSize_ = pageSize_ - reservedHeaderSize_;
    }
  }
  // auto lastHeaderIt = std::prev(headerEndIt);
  // size_ = lastHeaderIt->chunkSize;
}

FileBuffer::~FileBuffer() {
  // need to free pages
  // NOP
}

void FileBuffer::reserve(const size_t numBytes) {
  size_t numPagesRequested = (numBytes + pageSize_ - 1) / pageSize_;
  size_t numCurrentPages = multiPages_.size();
  int epoch = fm_->epoch();

  for (size_t pageNum = numCurrentPages; pageNum < numPagesRequested; ++pageNum) {
    Page page = addNewMultiPage(epoch);
    writeHeader(page, pageNum, epoch);
  }
}

void FileBuffer::calcHeaderBuffer() {
  // 3 * sizeof(int) is for headerSize, for pageId and versionEpoch
  // sizeof(size_t) is for chunkSize
  // reservedHeaderSize_ = (chunkKey_.size() + 3) * sizeof(int) + sizeof(size_t);
  reservedHeaderSize_ = (chunkKey_.size() + 3) * sizeof(int);
  size_t headerMod = reservedHeaderSize_ % headerBufferOffset_;
  if (headerMod > 0) {
    reservedHeaderSize_ += headerBufferOffset_ - headerMod;
  }
  // pageDataSize_ = pageSize_-reservedHeaderSize_;
}

void FileBuffer::freePages() {
  // Need to zero headers (actually just first four bytes of header)

  // First delete metadata pages
  for (auto metaPageIt = metadataPages_.pageVersions.begin();
       metaPageIt != metadataPages_.pageVersions.end();
       ++metaPageIt) {
    FileInfo* fileInfo = fm_->getFileInfoForFileId(metaPageIt->fileId);
    fileInfo->freePage(metaPageIt->pageNum);
  }

  // Now delete regular pages
  for (auto multiPageIt = multiPages_.begin(); multiPageIt != multiPages_.end();
       ++multiPageIt) {
    for (auto pageIt = multiPageIt->pageVersions.begin();
         pageIt != multiPageIt->pageVersions.end();
         ++pageIt) {
      FileInfo* fileInfo = fm_->getFileInfoForFileId(pageIt->fileId);
      fileInfo->freePage(pageIt->pageNum);
    }
  }
}

struct readThreadDS {
  FileMgr* t_fm;       // ptr to FileMgr
  size_t t_startPage;  // start page for the thread
  size_t t_endPage;    // last page for the thread
  int8_t* t_curPtr;    // pointer to the current location of the target for the thread
  size_t t_bytesLeft;  // number of bytes to be read in the thread
  size_t t_startPageOffset;  // offset - used for the first page of the buffer
  bool t_isFirstPage;        // true - for first page of the buffer, false - otherwise
  std::vector<MultiPage> multiPages;  // MultiPages of the FileBuffer passed to the thread
};

static size_t readForThread(FileBuffer* fileBuffer, const readThreadDS threadDS) {
  size_t startPage = threadDS.t_startPage;  // start reading at startPage, including it
  size_t endPage = threadDS.t_endPage;      // stop reading at endPage, not including it
  int8_t* curPtr = threadDS.t_curPtr;
  size_t bytesLeft = threadDS.t_bytesLeft;
  size_t totalBytesRead = 0;
  bool isFirstPage = threadDS.t_isFirstPage;

  // Traverse the logical pages
  for (size_t pageNum = startPage; pageNum < endPage; ++pageNum) {
    CHECK(threadDS.multiPages[pageNum].pageSize == fileBuffer->pageSize());
    Page page = threadDS.multiPages[pageNum].current();

    FileInfo* fileInfo = threadDS.t_fm->getFileInfoForFileId(page.fileId);
    CHECK(fileInfo);

    // Read the page into the destination (dst) buffer at its
    // current (cur) location
    size_t bytesRead = 0;
    if (isFirstPage) {
      bytesRead = fileInfo->read(
          page.pageNum * fileBuffer->pageSize() + threadDS.t_startPageOffset +
              fileBuffer->reservedHeaderSize(),
          min(fileBuffer->pageDataSize() - threadDS.t_startPageOffset, bytesLeft),
          curPtr);
      isFirstPage = false;
    } else {
      bytesRead = fileInfo->read(
          page.pageNum * fileBuffer->pageSize() + fileBuffer->reservedHeaderSize(),
          min(fileBuffer->pageDataSize(), bytesLeft),
          curPtr);
    }
    curPtr += bytesRead;
    bytesLeft -= bytesRead;
    totalBytesRead += bytesRead;
  }
  CHECK(bytesLeft == 0);

  return (totalBytesRead);
}

void FileBuffer::read(int8_t* const dst,
                      const size_t numBytes,
                      const size_t offset,
                      const MemoryLevel dstBufferType,
                      const int deviceId) {
  if (dstBufferType != CPU_LEVEL) {
    LOG(FATAL) << "Unsupported Buffer type";
  }

  // variable declarations
  size_t startPage = offset / pageDataSize_;
  size_t startPageOffset = offset % pageDataSize_;
  size_t numPagesToRead =
      (numBytes + startPageOffset + pageDataSize_ - 1) / pageDataSize_;
  /*
  if (startPage + numPagesToRead > multiPages_.size()) {
      cout << "Start page: " << startPage << endl;
      cout << "Num pages to read: " << numPagesToRead << endl;
      cout << "Num multipages: " << multiPages_.size() << endl;
      cout << "Offset: " << offset << endl;
      cout << "Num bytes: " << numBytes << endl;
  }
  */

  CHECK(startPage + numPagesToRead <= multiPages_.size());

  size_t numPagesPerThread = 0;
  size_t numBytesCurrent = numBytes;  // total number of bytes still to be read
  size_t bytesRead = 0;               // total number of bytes already being read
  size_t bytesLeftForThread = 0;      // number of bytes to be read in the thread
  size_t numExtraPages = 0;  // extra pages to be assigned one per thread as needed
  size_t numThreads = fm_->getNumReaderThreads();
  std::vector<readThreadDS>
      threadDSArr;  // array of threadDS, needed to avoid racing conditions

  if (numPagesToRead > numThreads) {
    numPagesPerThread = numPagesToRead / numThreads;
    numExtraPages = numPagesToRead - (numThreads * numPagesPerThread);
  } else {
    numThreads = numPagesToRead;
    numPagesPerThread = 1;
  }

  /* set threadDS for the first thread */
  readThreadDS threadDS;
  threadDS.t_fm = fm_;
  threadDS.t_startPage = offset / pageDataSize_;
  if (numExtraPages > 0) {
    threadDS.t_endPage = threadDS.t_startPage + numPagesPerThread + 1;
    numExtraPages--;
  } else {
    threadDS.t_endPage = threadDS.t_startPage + numPagesPerThread;
  }
  threadDS.t_curPtr = dst;
  threadDS.t_startPageOffset = offset % pageDataSize_;
  threadDS.t_isFirstPage = true;

  bytesLeftForThread = min(((threadDS.t_endPage - threadDS.t_startPage) * pageDataSize_ -
                            threadDS.t_startPageOffset),
                           numBytesCurrent);
  threadDS.t_bytesLeft = bytesLeftForThread;
  threadDS.multiPages = getMultiPage();

  if (numThreads == 1) {
    bytesRead += readForThread(this, threadDS);
  } else {
    std::vector<std::future<size_t>> threads;

    for (size_t i = 0; i < numThreads; i++) {
      threadDSArr.push_back(threadDS);
      threads.push_back(
          std::async(std::launch::async, readForThread, this, threadDSArr[i]));

      // calculate elements of threadDS
      threadDS.t_fm = fm_;
      threadDS.t_isFirstPage = false;
      threadDS.t_curPtr += bytesLeftForThread;
      threadDS.t_startPage +=
          threadDS.t_endPage -
          threadDS.t_startPage;  // based on # of pages read on previous iteration
      if (numExtraPages > 0) {
        threadDS.t_endPage = threadDS.t_startPage + numPagesPerThread + 1;
        numExtraPages--;
      } else {
        threadDS.t_endPage = threadDS.t_startPage + numPagesPerThread;
      }
      numBytesCurrent -= bytesLeftForThread;
      bytesLeftForThread = min(
          ((threadDS.t_endPage - threadDS.t_startPage) * pageDataSize_), numBytesCurrent);
      threadDS.t_bytesLeft = bytesLeftForThread;
      threadDS.multiPages = getMultiPage();
    }

    for (auto& p : threads) {
      p.wait();
    }
    for (auto& p : threads) {
      bytesRead += p.get();
    }
  }
  CHECK(bytesRead == numBytes);
}

void FileBuffer::copyPage(Page& srcPage,
                          Page& destPage,
                          const size_t numBytes,
                          const size_t offset) {
  // FILE *srcFile = fm_->files_[srcPage.fileId]->f;
  // FILE *destFile = fm_->files_[destPage.fileId]->f;
  CHECK(offset + numBytes < pageDataSize_);
  FileInfo* srcFileInfo = fm_->getFileInfoForFileId(srcPage.fileId);
  FileInfo* destFileInfo = fm_->getFileInfoForFileId(destPage.fileId);

  int8_t* buffer = new int8_t[numBytes];
  size_t bytesRead = srcFileInfo->read(
      srcPage.pageNum * pageSize_ + offset + reservedHeaderSize_, numBytes, buffer);
  CHECK(bytesRead == numBytes);
  size_t bytesWritten = destFileInfo->write(
      destPage.pageNum * pageSize_ + offset + reservedHeaderSize_, numBytes, buffer);
  CHECK(bytesWritten == numBytes);
  delete[] buffer;
}

Page FileBuffer::addNewMultiPage(const int epoch) {
  Page page = fm_->requestFreePage(pageSize_, false);
  MultiPage multiPage(pageSize_);
  multiPage.epochs.push_back(epoch);
  multiPage.pageVersions.push_back(page);
  multiPages_.push_back(multiPage);
  return page;
}

void FileBuffer::writeHeader(Page& page,
                             const int pageId,
                             const int epoch,
                             const bool writeMetadata) {
  int intHeaderSize = chunkKey_.size() + 3;  // does not include chunkSize
  vector<int> header(intHeaderSize);
  // in addition to chunkkey we need size of header, pageId, version
  header[0] =
      (intHeaderSize - 1) * sizeof(int);  // don't need to include size of headerSize
                                          // value - sizeof(size_t) is for chunkSize
  std::copy(chunkKey_.begin(), chunkKey_.end(), header.begin() + 1);
  header[intHeaderSize - 2] = pageId;
  header[intHeaderSize - 1] = epoch;
  FileInfo* fileInfo = fm_->getFileInfoForFileId(page.fileId);
  size_t pageSize = writeMetadata ? METADATA_PAGE_SIZE : pageSize_;
  fileInfo->write(
      page.pageNum * pageSize, (intHeaderSize) * sizeof(int), (int8_t*)&header[0]);
}

void FileBuffer::readMetadata(const Page& page) {
  FILE* f = fm_->getFileForFileId(page.fileId);
  fseek(f, page.pageNum * METADATA_PAGE_SIZE + reservedHeaderSize_, SEEK_SET);
  fread((int8_t*)&pageSize_, sizeof(size_t), 1, f);
  fread((int8_t*)&size_, sizeof(size_t), 1, f);
  vector<int> typeData(NUM_METADATA);  // assumes we will encode hasEncoder, bufferType,
                                       // encodingType, encodingBits all as int
  fread((int8_t*)&(typeData[0]), sizeof(int), typeData.size(), f);
  int version = typeData[0];
  CHECK(version == METADATA_VERSION);  // add backward compatibility code here
  hasEncoder = static_cast<bool>(typeData[1]);
  if (hasEncoder) {
    sqlType.set_type(static_cast<SQLTypes>(typeData[2]));
    sqlType.set_subtype(static_cast<SQLTypes>(typeData[3]));
    sqlType.set_dimension(typeData[4]);
    sqlType.set_scale(typeData[5]);
    sqlType.set_notnull(static_cast<bool>(typeData[6]));
    sqlType.set_compression(static_cast<EncodingType>(typeData[7]));
    sqlType.set_comp_param(typeData[8]);
    sqlType.set_size(typeData[9]);
    initEncoder(sqlType);
    encoder->readMetadata(f);
  }
}

void FileBuffer::writeMetadata(const int epoch) {
  // Right now stats page is size_ (in bytes), bufferType, encodingType,
  // encodingDataType, numElements
  Page page = fm_->requestFreePage(METADATA_PAGE_SIZE, true);
  writeHeader(page, -1, epoch, true);
  FILE* f = fm_->getFileForFileId(page.fileId);
  fseek(f, page.pageNum * METADATA_PAGE_SIZE + reservedHeaderSize_, SEEK_SET);
  fwrite((int8_t*)&pageSize_, sizeof(size_t), 1, f);
  fwrite((int8_t*)&size_, sizeof(size_t), 1, f);
  vector<int> typeData(NUM_METADATA);  // assumes we will encode hasEncoder, bufferType,
                                       // encodingType, encodingBits all as int
  typeData[0] = METADATA_VERSION;
  typeData[1] = static_cast<int>(hasEncoder);
  if (hasEncoder) {
    typeData[2] = static_cast<int>(sqlType.get_type());
    typeData[3] = static_cast<int>(sqlType.get_subtype());
    typeData[4] = sqlType.get_dimension();
    typeData[5] = sqlType.get_scale();
    typeData[6] = static_cast<int>(sqlType.get_notnull());
    typeData[7] = static_cast<int>(sqlType.get_compression());
    typeData[8] = sqlType.get_comp_param();
    typeData[9] = sqlType.get_size();
  }
  fwrite((int8_t*)&(typeData[0]), sizeof(int), typeData.size(), f);
  if (hasEncoder) {  // redundant
    encoder->writeMetadata(f);
  }
  metadataPages_.epochs.push_back(epoch);
  metadataPages_.pageVersions.push_back(page);
}

/*
void FileBuffer::checkpoint() {
    if (isAppended_) {
        Page page = multiPages_[multiPages.size()-1].current();
        writeHeader(page,0,multiPages_[0].epochs.back());
    }
    isDirty_ = false;
    isUpdated_ = false;
    isAppended_ = false;
}
*/

void FileBuffer::append(int8_t* src,
                        const size_t numBytes,
                        const MemoryLevel srcBufferType,
                        const int deviceId) {
  isDirty_ = true;
  isAppended_ = true;

  size_t startPage = size_ / pageDataSize_;
  size_t startPageOffset = size_ % pageDataSize_;
  size_t numPagesToWrite =
      (numBytes + startPageOffset + pageDataSize_ - 1) / pageDataSize_;
  size_t bytesLeft = numBytes;
  int8_t* curPtr = src;  // a pointer to the current location in dst being written to
  size_t initialNumPages = multiPages_.size();
  size_ = size_ + numBytes;
  int epoch = fm_->epoch();
  for (size_t pageNum = startPage; pageNum < startPage + numPagesToWrite; ++pageNum) {
    Page page;
    if (pageNum >= initialNumPages) {
      page = addNewMultiPage(epoch);
      writeHeader(page, pageNum, epoch);
    } else {
      // we already have a new page at current
      // epoch for this page - just grab this page
      page = multiPages_[pageNum].current();
    }
    CHECK(page.fileId >= 0);  // make sure page was initialized
    FileInfo* fileInfo = fm_->getFileInfoForFileId(page.fileId);
    size_t bytesWritten;
    if (pageNum == startPage) {
      bytesWritten = fileInfo->write(
          page.pageNum * pageSize_ + startPageOffset + reservedHeaderSize_,
          min(pageDataSize_ - startPageOffset, bytesLeft),
          curPtr);
    } else {
      bytesWritten = fileInfo->write(page.pageNum * pageSize_ + reservedHeaderSize_,
                                     min(pageDataSize_, bytesLeft),
                                     curPtr);
    }
    curPtr += bytesWritten;
    bytesLeft -= bytesWritten;
  }
  CHECK(bytesLeft == 0);
}

void FileBuffer::write(int8_t* src,
                       const size_t numBytes,
                       const size_t offset,
                       const MemoryLevel srcBufferType,
                       const int deviceId) {
  if (srcBufferType != CPU_LEVEL) {
    LOG(FATAL) << "Unsupported Buffer type";
  }
  isDirty_ = true;
  if (offset < size_) {
    isUpdated_ = true;
  }
  bool tempIsAppended = false;

  if (offset + numBytes > size_) {
    tempIsAppended = true;  // because isAppended_ could have already been true - to avoid
                            // rewriting header
    isAppended_ = true;
    size_ = offset + numBytes;
  }

  size_t startPage = offset / pageDataSize_;
  size_t startPageOffset = offset % pageDataSize_;
  size_t numPagesToWrite =
      (numBytes + startPageOffset + pageDataSize_ - 1) / pageDataSize_;
  size_t bytesLeft = numBytes;
  int8_t* curPtr = src;  // a pointer to the current location in dst being written to
  size_t initialNumPages = multiPages_.size();
  int epoch = fm_->epoch();

  if (startPage >
      initialNumPages) {  // means there is a gap we need to allocate pages for
    for (size_t pageNum = initialNumPages; pageNum < startPage; ++pageNum) {
      Page page = addNewMultiPage(epoch);
      writeHeader(page, pageNum, epoch);
    }
  }
  for (size_t pageNum = startPage; pageNum < startPage + numPagesToWrite; ++pageNum) {
    Page page;
    if (pageNum >= initialNumPages) {
      page = addNewMultiPage(epoch);
      writeHeader(page, pageNum, epoch);
    } else if (multiPages_[pageNum].epochs.back() <
               epoch) {  // need to create new page b/c this current one lags epoch and we
                         // can't overwrite it also need to copy if we are on first or
                         // last page
      Page lastPage = multiPages_[pageNum].current();
      page = fm_->requestFreePage(pageSize_, false);
      multiPages_[pageNum].epochs.push_back(epoch);
      multiPages_[pageNum].pageVersions.push_back(page);
      if (pageNum == startPage && startPageOffset > 0) {
        // copyPage takes care of header offset so don't worry
        // about it
        copyPage(lastPage, page, startPageOffset, 0);
      }
      if (pageNum == startPage + numPagesToWrite &&
          bytesLeft > 0) {  // bytesLeft should always > 0
        copyPage(lastPage,
                 page,
                 pageDataSize_ - bytesLeft,
                 bytesLeft);  // these would be empty if we're appending but we won't
                              // worry about it right now
      }
      writeHeader(page, pageNum, epoch);
    } else {
      // we already have a new page at current
      // epoch for this page - just grab this page
      page = multiPages_[pageNum].current();
    }
    CHECK(page.fileId >= 0);  // make sure page was initialized
    FileInfo* fileInfo = fm_->getFileInfoForFileId(page.fileId);
    size_t bytesWritten;
    if (pageNum == startPage) {
      bytesWritten = fileInfo->write(
          page.pageNum * pageSize_ + startPageOffset + reservedHeaderSize_,
          min(pageDataSize_ - startPageOffset, bytesLeft),
          curPtr);
    } else {
      bytesWritten = fileInfo->write(page.pageNum * pageSize_ + reservedHeaderSize_,
                                     min(pageDataSize_, bytesLeft),
                                     curPtr);
    }
    curPtr += bytesWritten;
    bytesLeft -= bytesWritten;
    if (tempIsAppended && pageNum == startPage + numPagesToWrite - 1) {  // if last page
      //@todo below can lead to undefined - we're overwriting num
      // bytes valid at checkpoint
      writeHeader(page, 0, multiPages_[0].epochs.back(), true);
    }
  }
  CHECK(bytesLeft == 0);
}

}  // namespace File_Namespace
