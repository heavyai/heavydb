/**
 * @file	Block.h
 * @author 	Steven Stewart
 *
 * This file contains the declaration and definition of a BlockAddr type and a BlockInfo type.
 */

#ifndef BLOCK_H_
#define BLOCK_H_

#include "../../Shared/types.h"

namespace File_Namespace {

/**
 * @type BlockAddr
 * @brief A logical block within a file.
 *
 * A BlockAddr object represents a logical block in a file managed by the file system. There
 * are two member variables that uniquely identify a block: a file identifier and an
 * address where the block begins in the containing file.
 */
struct BlockAddr {
	int fileId;					/**< the file identifier of the file containing the block */
	mapd_size_t addr;			/**< the address of the block within its containing file */
	unsigned int epoch;			/**< indicates the last temporal reference point for which the block is current */
	mapd_size_t endByteOffset;	/**< one position after the last used location in the block */

	/// Constructor
	BlockAddr(int fileId, mapd_size_t addr, unsigned int epoch = 0, mapd_size_t endByteOffset = 0) :
		fileId(fileId), addr(addr),	epoch(epoch), endByteOffset(endByteOffset) {}

	/// "Clears" the block by setting the endByteOffset to 0
	void clear() { endByteOffset = 0; }
};

/**
 * @type BlockInfo
 * @brief This struct wraps a set of block addresses with metadata associated with a logical block.
 *
 * This struct stores the locations (file id and block number) of the copies of the block. Each copy
 * has an epoch value, which is a temporal reference for when the copy was last updated.
 */
struct BlockInfo {
	std::vector<BlockAddr*> addr;
	mapd_size_t blockSize;		/**< the logical block size in bytes */

	/// Constructor
	BlockInfo(mapd_size_t blockSize, mapd_size_t order = 0) : blockSize(blockSize) {}

	/// Destructor
	~BlockInfo() {
		for (int i = 0; i < addr.size(); i++)
			delete addr[i];
		addr.clear();
	}
};

} // File_Namespace

#endif /* BLOCK_H_ */
