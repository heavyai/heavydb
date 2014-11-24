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
        assert (startPage + numPagesToRead <= pages_.size());
        // Traverse the logical pages
        mapd_size_t bytesLeft = numBytes;
        for (size_t pageNum = startPage; pageNum < startPage  + numPagesToRead; ++pageNum) {

            assert(pages_[pageNum].pageSize == pageSize_);
            Page b = pages_[pageNum].current();
            printf("read: fileId=%d pageNum=%lu pageSize=%lu\n", b.fileId, b.pageNum, pageSize_);

            // Open the file
            FILE *f = nullptr;
            auto fileIt = openFiles.find(b.fileId);
            if (fileIt == openFiles.end()) {
                f = File_Namespace::open(b.fileId);
                openFiles[b.fileId] = f;
            }
            else
                f = fileIt->second;
            assert(f);

            // Read the page into the destination (dst) buffer at its
            // current (cur) location
            size_t bytesRead;
            if (pageNum == startPage) 
                bytesRead = File_Namespace::read(f, b.pageNum * pageSize_ + startPageOffset, min(pageSize_ - startPageOffset,bytesLeft), cur);
            else 
                bytesRead = File_Namespace::read(f, b.pageNum * pageSize_, min(pageSize_,bytesLeft), cur);
            bytesLeft -= bytesRead
            cur += bytesRead
        }
        assert (bytesLeft == 0);

        // is this below thread safe? Maybe Instead let's just leave the files open
        // for now
        for (auto fileIt = openFiles.begin(); fileIt != openFiles.end(); ++fileIt)
            close(fileIt->second);
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
            if (pageNum >= pages_.size() ||  pages_[i].epochs.back() < epoch) {
                numNewPages++;
            }
        }
        fm_->requestFreeBlocks(numNewPages, pageSize_, freePages);


        



    

        
        `











