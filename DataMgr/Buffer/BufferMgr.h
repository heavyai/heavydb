/**
 * @file	BufferMgr.h
 * @author	Steven Stewart <steve@map-d.com>
 *
 * This file includes the class specification for the buffer manager (BufferMgr),
 * and related data structures and types.
 *
 */
#include <vector>
#include <string>
#include "../../Shared/types.h"

/**
 * @class 	BufferMgr
 * @author	Steven Stewart <steve@map-d.com>
 * @brief The buffer manager handles the caching and movement of data within the memory hierarchy (CPU/GPU).
 *
 * The buffer manager is the subsystem responsible for the allocation of the buffer space (also called the
 * memory pool or memory cache).
 *
 * Map-D uses a three-level storage hierarchy: nonvolatile (disk), main memory, and GPU memory. The buffer
 * manager handles the caching and movement of data across this hierarchy. The main goal of the buffer manager
 * is to maximize the chance that, when a block is accessed, it is already in the faster main or GPU memory,
 * such that no disk access is required.
 */
class BufferMgr {

public:
	/**
	 * A constructor that instantiates a buffer manager instance.
	 */
	BufferMgr();

	/**
	 * A destructor that cleans up resources used by a buffer manager instance.
	 */
	~BufferMgr();



private:

}; // BufferMgr




