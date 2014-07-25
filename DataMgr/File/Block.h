/**
 * @file	Block.h
 * @author 	Steven Stewart
 *
 * This file contains the declaration and definition of a Block type and a BlockInfo type.
 */

#ifndef _BLOCK_H_
#define _BLOCK_H_

#include <cassert>
#include <deque>	
#include "../../Shared/types.h"

namespace File_Namespace {

/**
 * @struct Block
 * @brief A logical block (Block) belongs to a file on disk.
 *
 * A Block struct stores the file id for the file it belongs to, and it
 * stores its beginning and ending locations within that file. Note that
 * the ending location signifies one position beyond the last used byte
 * in the block. To use a block properly, the owning file should specify
 * a block size "blockSize" such that: (end - begin) <= blockSize.
 */
struct Block {
	int fileId;			/// unique identifier of the owning file
	mapd_size_t begin;	/// offset of first byte from beginning of file
	mapd_size_t end;	/// last used byte is offset at end-1

	/// Constructor
	Block(int fileId, mapd_size_t begin) {
		this->fileId = fileId;
		this->begin = begin;
		this->end = begin; /// (begin >= end) => empty block
	}
};

/**
 * @struct MultiBlock
 * @brief The MultiBlock stores versions of the same logical block in a queue.
 *
 * The purpose of MultiBlock is to support storing multiple versions of the same
 * block, which may be located in different locations and in different files.
 * Associated with each version of a block is an "epoch" value, which is a temporal
 * reference.
 *
 *
 * Note that it should always be the case that version.size() == epoch.size().
 */
struct MultiBlock {
	int fileId;
	mapd_size_t blockSize;
	std::deque<Block*> version;
	std::deque<int> epoch;

	/// Constructor
	MultiBlock(int fileIdIn, mapd_size_t blockSizeIn) :
		fileId(fileIdIn), blockSize(blockSizeIn) {}

	/// Destructor -- purges all blocks
	~MultiBlock() {
		while (version.size() > 0)
			pop();
	}

	/// Returns a reference to the most recent version of the block (optionally, the epoch
	/// is returned via the parameter "epoch").
	inline Block& current(int *epoch = NULL) {
		assert(version.size() > 0);
		if (epoch != NULL)
			*epoch = this->epoch.back();
		return *version.back();
	}

	/// Pushes a new block with epoch value
	inline void push(Block *b, int epoch) {
		version.push_back(b);
		this->epoch.push_back(epoch);
	}

	/// Purges the oldest block
	inline void pop() {
		delete version.front(); // frees memory used by oldest Block
		version.pop_front();
		this->epoch.pop_front();
	}
};

} // File_Namespace

#endif /* _BLOCK_H_ */
