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
	mapd_size_t blockNum;	/// beginning byte offset of block within file
	mapd_size_t used;		/// ending of block (1 byte beyond the last used byte)

	/// Constructor
	Block(int fileId, mapd_size_t blockNum) {
		this->fileId = fileId;
		this->blockNum = blockNum;
		this->used = 0;
	}
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
	std::deque<Block*> version;
	std::deque<int> epoch;

	/// Constructor
	MultiBlock(mapd_size_t blockSizeIn) :
		blockSize(blockSizeIn) {}

	/// Destructor -- purges all blocks
	~MultiBlock() {
		while (version.size() > 0)
			pop();
	}

	/// Returns a reference to the most recent version of the block (optionally, the epoch
	/// is returned via the parameter "epoch").
	inline Block& current(int *epoch = NULL) {
		assert(version.size() > 0); // @todo should use proper exception handling
		if (epoch != NULL)
			*epoch = this->epoch.back();
		return *version.back();
	}

	/// Pushes a new block with epoch value
	inline void push(Block *b, int epoch) {
		version.push_back(b);
		this->epoch.push_back(epoch);
		assert(this->version.size() == this->epoch.size());
	}

	/// Purges the oldest block
	inline void pop() {
		delete version.front(); // frees memory used by oldest Block
		version.pop_front();
		this->epoch.pop_front();
		assert(this->version.size() == this->epoch.size());
	}
};

} // File_Namespace

#endif // DATAMGR_FILE_BLOCK_H
