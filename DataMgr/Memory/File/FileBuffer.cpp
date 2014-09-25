/**
 * @file        FileBuffer.cpp
 * @author      Steven Stewart <steve@map-d.com>
 */
#include "FileBuffer.h"
#include "File.h"
#include "FileMgr.h"
#include <map>

namespace File_Namespace {
    
    FileBuffer::FileBuffer(mapd_size_t pageSize, FileMgr *fm) : blockSize_(pageSize), fm_(fm) {
        assert(fm_);
        // NOP
    }
    
    FileBuffer::~FileBuffer() {
        // NOP
    }

    void FileBuffer::read(mapd_addr_t const dst, const mapd_size_t offset, const mapd_size_t nbytes) {
        // variable declarations
        size_t nblocks;     // the number of logical blocks in this FileBuffer
        mapd_addr_t cur;    // a pointer to the current location in dst being written to
        std::map<int, FILE*> openFiles; // keeps track of opened files
        mapd_size_t nblocksToRead; // the number of blocks to be read
        
        // initialize variables
        nblocks = blocks_.size();
        cur = dst + offset;
        nblocksToRead = (nbytes + blockSize_ - 1) / blockSize_;
        
        // Traverse the logical blocks
        for (size_t i = 0; i < nblocksToRead; ++i) {
            assert(blocks_[i].blockSize == blockSize_);
            
            // Obtain the most recent version of the current block
            Block b = blocks_[i].current();
            printf("read: fileId=%d blockNum=%lu blockSize=%lu\n", b.fileId, b.blockNum, blockSize_);
            
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
            
            // Read the block into the destination (dst) buffer at its
            // current (cur) location
            // size_t bytesRead = File_Namespace::readBlock(f, blockSize_, b.blockNum, cur);
            size_t bytesRead = File_Namespace::read(f, i * blockSize_, blockSize_, cur);
            // assert(bytesRead == blockSize_);
            
            // testing
            mapd_byte_t tmp[bytesRead];
            File_Namespace::read(f, b.blockNum * blockSize_, bytesRead, tmp);
            int *p_tmp = (int*)&tmp;
            for (int i = 0; i < blockSize_; ++i) {
                printf("read: %d\n", p_tmp[i]);
            }
            
            cur += blockSize_;
        }
        
        // Close any open files
        for (auto fileIt = openFiles.begin(); fileIt != openFiles.end(); ++fileIt)
            close(fileIt->second);
    }

    void FileBuffer::write(mapd_addr_t src, const mapd_size_t offset, const mapd_size_t nbytes) {
        // variable declarations
        mapd_size_t bytesToWrite;           // number of bytes remaining to be written
        mapd_size_t numBlocks;              // number of blocks to be written
        mapd_size_t numNewBlocks;           // number of free blocks to request from file manager
        std::vector<Block> freeBlocks;      // new blocks to be appended to this buffer
        
        // initializations
        bytesToWrite = nbytes;
        numBlocks = (offset + nbytes + blockSize_ - 1) / blockSize_;
        
        // determine how many new blocks are needed
        numNewBlocks = 0;
        for (mapd_size_t i = 0; i < blocks_.size() && i < numBlocks; ++i)
            if (blocks_[i].epochs.back() < fm_->epoch())
                numNewBlocks++;
        numNewBlocks += numBlocks - numNewBlocks; // add number of blocks to append
        
        // request the new blocks from the file manager
        fm_->requestFreeBlocks(numNewBlocks, blockSize_, freeBlocks);

        // append the new blocks
        for (auto blkIt = freeBlocks.begin(); blkIt != freeBlocks.end(); ++blkIt) {
            Block b = *blkIt;
            MultiBlock mb(blockSize_);
            mb.epochs.push_back(fm_->epoch());
            mb.blkVersions.push_back(b);
            blocks_.push_back(mb);
        }
        freeBlocks.clear();

        // write
        mapd_size_t curOffset = offset;
        int j = 0;
        for (int i = 0; i < blocks_.size() && i < numBlocks; i++) {
            if (((i+1) * blockSize_) < offset)
                continue;
            
            // determine number of bytes to write to current block
            mapd_size_t curBytesToWrite = std::min(bytesToWrite, blockSize_ - (curOffset % blockSize_));
            assert(curBytesToWrite <= blockSize_);
            
            // write to the block
            Block b = blocks_[j].current();
            FILE *f = File_Namespace::open(b.fileId);
            File_Namespace::write(f, b.blockNum, curBytesToWrite, src + curOffset);
            File_Namespace::close(f);
            printf("write: fileId=%d blockNum=%lu curBytesToWrite=%lu\n", b.fileId, b.blockNum, curBytesToWrite);
            
            
            // testing
            f = File_Namespace::open(b.fileId);
            mapd_byte_t tmp[curBytesToWrite];
            File_Namespace::read(f, b.blockNum * blockSize_, curBytesToWrite, tmp);
            int *p_tmp = (int*)&tmp;
            for (int i = 0; i < blockSize_; ++i) {
                printf("wrote: %d\n", p_tmp[i]);
            }
            
            
            // update counters
            curOffset += curBytesToWrite;
            bytesToWrite -= curBytesToWrite;
            j++;
        }
    }

    void FileBuffer::append(mapd_addr_t src, const mapd_size_t nbytes) {
        
    }

    /// Returns the number of pages in the FileBuffer.
    mapd_size_t FileBuffer::pageCount() const {
        return 0;
    }
    
    /// Returns the size in bytes of each page in the FileBuffer.
    mapd_size_t FileBuffer::pageSize() const {
        return blockSize_;
    }
    
    /// Returns the total number of bytes allocated for the FileBuffer.
    mapd_size_t FileBuffer::size() const {
        return 0;
    }
    
    /// Returns the total number of used bytes in the FileBuffer.
    mapd_size_t FileBuffer::used() const {
        return 0;
    }
    
    /// Returns whether or not the FileBuffer has been modified since the last flush/checkpoint.
    bool FileBuffer::isDirty() const {
        return 0;
    }
    
} // File_Namespace
