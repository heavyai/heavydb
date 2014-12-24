/**
 * @file        FileBuffer.cpp
 * @author      Steven Stewart <steve@map-d.com>
 * @author      Todd Mostak <todd@map-d.com>
 */

#include "FileBuffer.h"
#include "File.h"
#include "FileMgr.h"
#include <map>

using namespace std;

namespace File_Namespace {
    mapd_size_t FileBuffer::headerBufferOffset_ = 32;

    FileBuffer::FileBuffer(FileMgr *fm, const mapd_size_t pageSize, const ChunkKey &chunkKey, const mapd_size_t numBytes) : fm_(fm), pageSize_(pageSize), chunkKey_(chunkKey), isDirty_(false) {
        // Create a new FileBuffer
        assert(fm_);
        calcHeaderBuffer();
        if (numBytes > 0) {
            // should expand to numBytes bytes
            size_t initialNumPages = (numBytes + pageSize_ -1) / pageSize_;
            int epoch = fm_-> epoch();
            for (size_t pageNum = 0; pageNum < initialNumPages; ++pageNum) {
                Page page = addNewMultiPage(epoch);
                writeHeader(page,pageNum,epoch);
            }
        }
    }

    FileBuffer::FileBuffer(FileMgr *fm, const mapd_size_t pageSize, const ChunkKey &chunkKey, const std::vector<HeaderInfo>::const_iterator &headerStartIt, const std::vector<HeaderInfo>::const_iterator &headerEndIt): fm_(fm), pageSize_(pageSize),chunkKey_(chunkKey),isDirty_(false) {
        // We are being assigned an existing FileBuffer on disk

        assert(fm_);
        calcHeaderBuffer();
        MultiPage multiPage(pageSize_);
        multiPages_.push_back(multiPage);
        //vector <int> pageAndVersionId = {-1,-1};
        int lastPageId = -1;
        //for (auto vecIt = headerVec.begin(); vecIt != headerVec.end(); ++vecIt) {
        for (auto vecIt = headerStartIt; vecIt != headerEndIt; ++vecIt) {
            int curPageId = vecIt -> pageId;
            if (curPageId != lastPageId) {
                assert (curPageId == lastPageId + 1);
                MultiPage multiPage(pageSize_);
                multiPages_.push_back(multiPage);
                lastPageId = curPageId;
            }
            multiPages_.back().epochs.push_back(vecIt -> versionEpoch);
            multiPages_.back().pageVersions.push_back(vecIt -> page);
        }
    }

    FileBuffer::~FileBuffer() {
        // need to free pages
        // NOP
    }

    void FileBuffer::reserve(const size_t numBytes) {
        size_t numPagesRequested = (numBytes + pageSize_ -1) / pageSize_;
        size_t numCurrentPages = multiPages_.size();
        int epoch = fm_-> epoch();

        for (size_t pageNum = numCurrentPages; pageNum < numPagesRequested; ++pageNum) {
            Page page = addNewMultiPage(epoch);
            writeHeader(page,pageNum,epoch);
        }
    }

    void FileBuffer::calcHeaderBuffer() {
        // 3 is for headerSize, for pageId and versionEpoch
        mapd_size_t headerSize = chunkKey_.size() * sizeof(int) + 3 * sizeof(int);
        reservedHeaderSize_ = headerSize;
        mapd_size_t headerMod = headerSize % headerBufferOffset_;
        if (headerMod > 0) {
            reservedHeaderSize_ += headerBufferOffset_ - reservedHeaderSize_;
        }
        pageDataSize_ = pageSize_-reservedHeaderSize_;
    }

    void FileBuffer::freePages() {
        // Need to zero headers (actually just first four bytes of header)
        int zeroVal = 0;
        mapd_addr_t zeroAddr = mapd_addr_t (&zeroVal);
        for (auto multiPageIt = multiPages_.begin(); multiPageIt != multiPages_.end(); ++multiPageIt) {
            for (auto pageIt = multiPageIt -> pageVersions.begin(); pageIt != multiPageIt -> pageVersions.end(); ++pageIt) { 
                FILE *f = fm_ -> getFileForFileId(pageIt -> fileId);
                File_Namespace::write(f,pageIt -> pageNum * pageSize_,sizeof(int),zeroAddr);
            }
        }
    }


    void FileBuffer::read(mapd_addr_t const dst, const mapd_size_t numBytes, const mapd_size_t offset) {
        // variable declarations
        mapd_addr_t curPtr = dst;    // a pointer to the current location in dst being written to
        mapd_size_t startPage = offset / pageDataSize_;
        mapd_size_t startPageOffset = offset % pageDataSize_;
        mapd_size_t numPagesToRead = (numBytes + startPageOffset + pageDataSize_ - 1) / pageDataSize_;
        //cout << "Start Page: " << startPage << endl;
        //cout << "Num pages To Read: " << numPagesToRead << endl;
        //cout << "Num pages existing: " << multiPages_.size() << endl;
        assert (startPage + numPagesToRead <= multiPages_.size());
        mapd_size_t bytesLeft = numBytes;

        // Traverse the logical pages
        for (size_t pageNum = startPage; pageNum < startPage  + numPagesToRead; ++pageNum) {

            assert(multiPages_[pageNum].pageSize == pageSize_);
            Page page = multiPages_[pageNum].current();
            //printf("read: fileId=%d pageNum=%lu pageSize=%lu\n", page.fileId, page.pageNum, pageDataSize_);

            //FILE *f = fm_ -> files_[page.fileId] -> f;
            FILE *f = fm_ -> getFileForFileId(page.fileId);
            assert(f);

            // Read the page into the destination (dst) buffer at its
            // current (cur) location
            size_t bytesRead;
            if (pageNum == startPage) {
                bytesRead = File_Namespace::read(f, page.pageNum * pageSize_ + startPageOffset + reservedHeaderSize_, min(pageDataSize_ - startPageOffset,bytesLeft), curPtr);
            }
            else {
                bytesRead = File_Namespace::read(f, page.pageNum * pageSize_ + reservedHeaderSize_, min(pageDataSize_,bytesLeft), curPtr);
            }
            curPtr += bytesRead;
            bytesLeft -= bytesRead;
        }
        assert (bytesLeft == 0);
    }

    void FileBuffer::copyPage(Page &srcPage, Page &destPage, const mapd_size_t numBytes, const mapd_size_t offset) { 
        //FILE *srcFile = fm_ -> files_[srcPage.fileId] -> f;
        //FILE *destFile = fm_ -> files_[destPage.fileId] -> f;
        assert(offset + numBytes < pageDataSize_);
        FILE *srcFile = fm_ -> getFileForFileId(srcPage.fileId); 
        FILE *destFile = fm_ -> getFileForFileId(destPage.fileId); 
        mapd_addr_t buffer = new mapd_byte_t [numBytes]; 
        size_t bytesRead = File_Namespace::read(srcFile,srcPage.pageNum * pageSize_ + offset+reservedHeaderSize_, numBytes, buffer);
        assert(bytesRead == numBytes);
        size_t bytesWritten = File_Namespace::write(destFile,destPage.pageNum * pageSize_ + offset + reservedHeaderSize_, numBytes, buffer);
        assert(bytesWritten == numBytes);
        delete [] buffer;
    }

    Page FileBuffer::addNewMultiPage(const int epoch) {
        Page page = fm_ -> requestFreePage(pageSize_);
        MultiPage multiPage(pageSize_);
        multiPage.epochs.push_back(epoch);
        multiPage.pageVersions.push_back(page);
        multiPages_.push_back(multiPage);
        return page;
    }

    void FileBuffer::writeHeader(Page &page, const int pageId, const int epoch) {
        int headerSize = chunkKey_.size() + 3;
        vector <int> header (headerSize);
        // in addition to chunkkey we need size of header, pageId, version
        header[0] = (headerSize - 1) * sizeof(int); // don't need to include size of headerSize value
        std::copy(chunkKey_.begin(), chunkKey_.end(), header.begin() + 1);
        header[headerSize-2] = pageId;
        header[headerSize-1] = epoch;
        FILE *f = fm_ -> getFileForFileId(page.fileId);
        File_Namespace::write(f, page.pageNum*pageSize_,headerSize * sizeof(int),(mapd_addr_t)&header[0]);
    }

    void FileBuffer::write(mapd_addr_t src,  const mapd_size_t numBytes, const mapd_size_t offset) {
        isDirty_ = true;
        mapd_size_t startPage = offset / pageDataSize_;
        mapd_size_t startPageOffset = offset % pageDataSize_;
        mapd_size_t numPagesToWrite = (numBytes + startPageOffset + pageDataSize_ - 1) / pageDataSize_; 
        mapd_size_t bytesLeft = numBytes;
        mapd_addr_t curPtr = src;    // a pointer to the current location in dst being written to
        mapd_size_t initialNumPages = multiPages_.size();
        int epoch = fm_-> epoch();

        if (startPage > initialNumPages) { // means there is a gap we need to allocate pages for
            for (size_t pageNum = initialNumPages; pageNum < startPage; ++pageNum) {
                Page page = addNewMultiPage(epoch);
                writeHeader(page,pageNum,epoch);
            }
        }
        for (size_t pageNum = startPage; pageNum < startPage  + numPagesToWrite; ++pageNum) {
            Page page;
            if (pageNum >= initialNumPages) {
                page = addNewMultiPage(epoch);
                writeHeader(page,pageNum,epoch);
            }
            else if (multiPages_[pageNum].epochs.back() < epoch) { // need to create new page b/c this current one lags epoch and we can't overwrite it 
    // also need to copy if we are on first or last page
                Page lastPage = multiPages_[pageNum].current();
                page = fm_ -> requestFreePage(pageSize_);
                multiPages_[pageNum].epochs.push_back(epoch);
                multiPages_[pageNum].pageVersions.push_back(page);
                if (pageNum == startPage && startPageOffset > 0) {
                    // copyPage takes care of header offset so don't worry
                    // about it
                    copyPage(lastPage,page,startPageOffset,0);
                }
                if (pageNum == startPage + numPagesToWrite && bytesLeft > 0) { // bytesLeft should always > 0
                    copyPage(lastPage,page,pageDataSize_-bytesLeft,bytesLeft); // these would be empty if we're appending but we won't worry about it right now
                }
                writeHeader(page,pageNum,epoch);
            }
            else {
                // we already have a new page at current
                // epoch for this page - just grab this page
                page = multiPages_[pageNum].current();
            }
            //cout << "Page: " << page.fileId << " " << page.pageNum << endl;
            assert(page.fileId >= 0); // make sure page was initialized
            FILE *f = fm_ -> getFileForFileId(page.fileId);
            size_t bytesWritten;
            if (pageNum == startPage) {
                bytesWritten = File_Namespace::write(f,page.pageNum*pageSize_ + startPageOffset + reservedHeaderSize_, min (pageDataSize_ - startPageOffset,bytesLeft),curPtr);
            }
            else {
                bytesWritten = File_Namespace::write(f, page.pageNum * pageSize_+reservedHeaderSize_, min(pageDataSize_,bytesLeft), curPtr);
            }
            curPtr += bytesWritten;
            bytesLeft -= bytesWritten;
        }
        assert (bytesLeft == 0);
    }



} // File_Namespace

