/**
 * @file	Block.h
 * @author 	Steven Stewart <steve@map-d.com>
 * @author	Gil Walzer <gil@map-d.com>
 * This file contains the declaration and definition of a Block type and a MultiBlock type.
 */

#ifndef DATAMGR_FILE_BLOCK_H
#define DATAMGR_FILE_BLOCK_H

#include <cassert>
#include <deque>
#include "../../Shared/types.h"

namespace File_Namespace {
    
    /**
     * @struct Block
     * @brief A logical block (Block) belongs to a file on disk.
     *
     * A Block struct stores the file id for the file it belongs to, and it
     * stores its block number and number of used bytes within the block.
     *
     * Note: the number of used bytes should not be greater than the block
     * size. The block size is determined by the containing file.
     */
    struct Block {
        int fileId;				/// unique identifier of the owning file
        mapd_size_t blockNum;	/// block number
        mapd_size_t used;		/// number of used bytes within the block
        
        /// Constructor
        Block(int fileId, mapd_size_t blockNum) : fileId(fileId), blockNum(blockNum), used(0) {}
    };
    
    /**
     * @struct MultiBlock
     * @brief The MultiBlock stores versions of the same logical block in a deque.
     *
     * The purpose of MultiBlock is to support storing multiple versions of the same
     * block, which may be located in different locations and in different files.
     * Associated with each version of a block is an "epoch" value, which is a temporal
     * reference.
     *
     * Note that it should always be the case that version.size() == epoch.size().
     */
    struct MultiBlock {
        mapd_size_t blockSize;
        std::deque<Block*> blkVersions;
        std::deque<int> epochs;
        
        /// Constructor
        MultiBlock(mapd_size_t blockSizeIn) :
		blockSize(blockSizeIn) {}
        
        /// Destructor -- purges all blocks
        ~MultiBlock() {
            while (blkVersions.size() > 0)
                pop();
        }
        
        /// Returns a reference to the most recent version of the block (optionally, the epoch
        /// is returned via the parameter "epoch").
        inline Block* current(int *epoch = NULL) {
            if (blkVersions.size() < 1)
                throw std::runtime_error("No current version of the block exists in this MultiBlock.");
            assert(blkVersions.size() > 0); // @todo should use proper exception handling
            if (epoch != NULL)
                *epoch = this->epochs.back();
            return blkVersions.back();
        }
        
        /// Pushes a new block with epoch value
        inline void push(const int fileId, const mapd_size_t blockNum, const int epoch) {
            blkVersions.push_back(new Block(fileId, blockNum));
            this->epochs.push_back(epoch);
            assert(this->blkVersions.size() == this->epochs.size());
        }
        
        /// Purges the oldest block
        inline void pop() {
            if (blkVersions.size() < 1)
                throw std::runtime_error("No block to pop.");
            delete blkVersions.front(); // frees memory used by oldest Block
            blkVersions.pop_front();
            this->epochs.pop_front();
            assert(this->blkVersions.size() == this->epochs.size());
        }
    };
    
} // File_Namespace

#endif // DATAMGR_FILE_BLOCK_H
