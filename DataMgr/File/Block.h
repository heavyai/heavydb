/**
 * @file	Block.h
 * @author 	Steven Stewart
 *
 */

#ifndef BLOCK_H_
#define BLOCK_H_

#include "../../Shared/types.h"

/**
 * @type BlockAddr
 * @brief A logical block within a file.
 *
 * A BlockAddr object represents a logical block in a file managed by the file system. There
 * are two member variables that uniquely identify a block: a file identifier and an
 * address where the block begins in the containing file.
 *
 * Additionally, a BlockAddr has certain metadata associated with it that is relevant to
 * the file manager. These are epoch (a temporal reference based on the last update of the
 * block, usually associated with a checkpoint), and endByteOffset, which is the address
 * of the last byte within the block.
 *
 * The BlockAddr type is essentially used to uniquely identify the location of blocks on
 * disk (literally, the block's address), but also to keep track of the "version" of the block,
 * because the file manager permits multiple copies of a block to exist, using the epoch value
 * in order to identify the most recent version of the block. This functionality is useful
 * in the context of checkpointing behaviour, which is an optimization used in the DBMS for
 * managing when disk blocks are flushed to disk in coordination with the logging mechanism.
 *
 * The comparison operators are overridden to so that BlockAddr instances are sorted based on
 * epoch value when stored within an ordered STL container such as a set.
 */
struct BlockAddr {
	int fileId;					/**< the file identifier of the file containing the block */
	mapd_size_t addr;			/**< the address of the block within its containing file */

	// metadata
	unsigned int epoch;			/**< indicates the last temporal reference point for which the block is current */
	mapd_size_t endByteOffset;	/**< the last address of the last byte written to the block */

	/// Constructor
	BlockAddr(int fileId, mapd_size_t addr, unsigned int epoch = 0, mapd_size_t endByteOffset = 0) {
		this->fileId = fileId;
		this->addr = addr;
		this->epoch = epoch;
		this->endByteOffset = endByteOffset;
	}

	/// used by find() for ordered containers
	bool operator == (const BlockAddr& item) const {
		return (item.epoch == this->epoch);
	}

	/// used as a sort predicate for ordered containers
	bool operator < (const BlockAddr& item) const {
		return (this->epoch < item.epoch);
	}
};

/**
 * @type BlockInfo
 * @brief This struct wraps a set of block addresses with metadata associated with a logical block.
 *
 * This struct stores the locations of the copies of the block. Each copy has an epoch associated
 * with it (@see BlockAddr), and this is accomplished by having a set of BlockAddr, which are
 * ordered by epoch.
 *
 * The BlockInfo object essentially represents a single logical block, and therefore has a block size
 * and an option order (set to 0 by default). The file manager defines a chunk as being an ordered
 * set of logical blocks (BlockInfo objects), and so the order variable is necessary in that context.
 */
struct BlockInfo {
	std::vector<BlockAddr> addr;

	// metadata about a logical block
	mapd_size_t blockSize;		/**< the logical block size in bytes */
	mapd_size_t order;			/**< used for sorting blocks */

	/// Constructor
	BlockInfo(mapd_size_t blockSize, mapd_size_t order = 0) :
		blockSize(blockSize), order(order) {}

	/// used by find() for ordered containers
	bool operator == (const BlockInfo& item) const {
		return (item.order == this->order);
	}

	/// used as a sort predicate for ordered containers
	bool operator < (const BlockInfo& item) const {
		return (this->order < item.order);
	}
};

#endif /* BLOCK_H_ */
