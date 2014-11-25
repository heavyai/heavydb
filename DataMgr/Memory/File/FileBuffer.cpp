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

    FileBuffer::FileBuffer(mapd_size_t pageSize, FileMgr *fm) : pageSize_(pageSize), fm_(fm) {
        assert(fm_);
        // NOP
    }

    FileBuffer::~FileBuffer() {
        // need to free pages
        // NOP
    }
    void FileBuffer::read(mapd_addr_t const dst, const mapd_size_t numBytes, const mapd_size_t offset) {
        // variable declarations
        size_t numPages;     // the number of logical pages in this FileBuffer
        mapd_addr_t cur;    // a pointer to the current location in dst being written to
        std::map<int, FILE*> openFiles; // keeps track of opened files
        mapd_size_t numPagesToRead; // the number of pages to be read

        // initialize variables
        // need to take into account offset
        cur = dst;
        mapd_size_t startPage = offset / pageSize_;
        mapd_size_t startPageOffset = offset % pageSize_;
        numPagesToRead = (numBytes + startPageOffset + pageSize_ - 1) / pageSize_;
        assert (startPage + numPagesToRead <= multiPages_.size());
        // Traverse the logical pages
        mapd_size_t bytesLeft = numBytes;
        for (size_t pageNum = startPage; pageNum < startPage  + numPagesToRead; ++pageNum) {

            assert(multiPages_[pageNum].pageSize == pageSize_);
            Page b = multiPages_[pageNum].current();
            printf("read: fileId=%d pageNum=%lu pageSize=%lu\n", b.fileId, b.pageNum, pageSize_);

            FILE *f = fm_ -> files_[b.fileId] -> f;
            assert(f);

            // Read the page into the destination (dst) buffer at its
            // current (cur) location
            size_t bytesRead;
            if (pageNum == startPage) 
                bytesRead = File_Namespace::read(f, b.pageNum * pageSize_ + startPageOffset, min(pageSize_ - startPageOffset,bytesLeft), cur);
            else 
                bytesRead = File_Namespace::read(f, b.pageNum * pageSize_, min(pageSize_,bytesLeft), cur);
            bytesLeft -= bytesRead;
            cur += bytesRead;
        }
        assert (bytesLeft == 0);

    }

    void FileBuffer::copyPage(Block srcPage, Block destPage, const mapd_size_t numBytes, const mapd_size_t offset) { 
        FILE *srcFile = fm_ -> files_[srcPage.fileId] -> f;
        FILE *destFile = fm_ -> files_[destPage.fileId] -> f;
        mapd_addr_t buffer = new mapd_byte_t [numBytes]; 
        size_t bytesRead = File_Namespace::read(srcFile,srcPage.pageNum * pageSize_ + offset, buffer);
        assert(bytesRead == numBytes);
        size_t bytesWritten = File_Namespace::write(destFile,destPage.pageNum * pageSize_ + offset,buffer);
        assert(bytesWritten == numBytes);
        delete [] buffer;
    }


    void FileBuffer::write(mapd_addr_t src,  const mapd_size_t numBytes, const mapd_size_t offset) {
        mapd_size_t bytesToWrite = numBytes;           // number of bytes remaining to be written
        mapd_size_t startPage = offset / pageSize_;
        mapd_size_t startPageOffset = offset % pageSize_;
        mapd_size_t numPagesToWrite = (numBytes + startPageOffset + pageSize_ - 1) / pageSize_; 
        mapd_size_t numNewPages;           // number of free pages to request from file manager
        std::vector<Page> freePages;      // new pages to be appended to this buffer
        mapd_size_t bytesLeft = numBytes;
        int epoch = fm_-> epoch();
        mapd_size_t numNewPages = 0;
        for (size_t pageNum = startPage; pageNum < startPage  + numPagesToWrite; ++pageNum) {
            Block *page = nullptr;
            if (pageNum >= multiPages_.size() ||  multiPages_[pageNum].epochs.back() < epoch) {
                page = fm_ -> requestFreeBlock(pageSize_);
            }
            else {
                page = multiPages_[pageNum].current();
            }
            // now determine if we need to copy this page?
            if (pageNum == startPage && startPageOffset > 0) {
                //copy(
        }
    }
