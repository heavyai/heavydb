/**
 * @file	BufferMgr.h
 * @author	Steven Stewart <steve@map-d.com>
 *
 * This file includes the class specification for the buffer manager (BufferMgr), and related
 * data structures and types.
 *
 * The database buffer (managed by the buffer manager) has to be maintained for interfacing
 * memory and disk. Map-D manages its buffer pool in the user address space, and the
 * database is divided into frames of equal size, and the buffer consists of "pages" that can
 * span 1 or more frames. The number of buffer frames can be specified as a DBMS parameter,
 * which remains constant during a DBMS session. The main goal of the buffer manager is the
 * minimization of physical I/O.
 *
 * For more details on the buffer manager class and its related types and data structure,
 * read the documentation below within this header file.
 */
#include <vector>
#include <string>
#include <cassert>
#include "../../Shared/types.h"
#include "../../Shared/errors.h"

struct Frame {
    mapd_size_t addr;
    bool isDirty;
};

/**
 * @brief PageBounds provides support for various page sizes.
 *
 * The PageBounds refers to the beginning and ending frame IDs over which
 * a page spans. In other words, a PageBounds of (10, 12) means that the
 * page occupies frames 10, 11, and 12. This functionality enables pages
 * to span multiple frames and to have varying sizes.
 */
typedef std::pair <std::mapd_size_t, std::mapd_size_t> PageBounds;

/**
 * A PageInfo struct contains the bounds of the page (beginning and ending
 * frame IDs) and metadata about the page. Although a page's properties are
 * publicly exposed, helper methods are provided for convenience.
 */
struct PageInfo {
    PageBounds bounds;  /**< the span of frame IDs occupied by this page */
    int pinCount;       /**< the number of pins (resources using the page) */
    bool dirty;         /**< indicates the page has been altered and differs from disk */

    /**< Increments the pin count for this page. */
    inline void pin() { pinCount++; }
    
    /**< Decrements the pin count for this page. */
    inline void unpin() { pinCount--; }     
    
    /**< Returns true if the page is dirty. */
    inline bool isDirty() { return dirty; } 
    
    /**< Returns the first frame ID occupied by the page. */
    inline mapd_size_t begin() { return bounds.first; }
    
    /**< Returns the last frame ID occupied by the page. */
    inline mapd_size_t end() { return bounds.second; }
    
    /**< Returns the number of frames occupied by the page. */
    inline mapd_size_t numFrames() { return end() - begin(); }
};

/**
 * @brief A ChunkToPageMap maps a chunk key to a page in the buffer pool.
 *
 * Chunks are uniquely identifiable by a chunk key (ChunkKey). When a chunk is brought
 * into the buffer pool, it occupies a contiguous set of frames, whose bounds and metadata
 * are encapsulated by a PageInfo struct. If the chunk currently exists in the buffer pool,
 * then the ChunkToPageMap maps the chunk, via its key, to its page in the buffer pool.
 */
typedef std::map<ChunkKey, PageInfo> ChunkToPageMap;

/**
 * @class 	BufferMgr
 * @author	Steven Stewart <steve@map-d.com>
 * @brief The buffer manager handles the caching and movement of data within the memory hierarchy (CPU/GPU).
 *
 * The buffer manager is the subsystem responsible for the allocation of the buffer space (also called the
 * memory pool or memory cache). 
 *
 * Map-D uses a three-level storage hierarchy: nonvolatile (disk), main memory, and GPU memory. The buffer
 * manager handles the caching and movement of data across this hierarchy. 
 *
 * In general, the buffer manager serves as an intermediary between DBMS modules and the memory subsystems.
 * The main goal of the buffer manager is to maximize the chance that, when a block is accessed, no disk
 * access is required. This is accomplished by caching blocks in main and/or GPU memory.
 *
 */
class BufferMgr {

public:
    /**
     * @brief A constructor that instantiates a buffer manager instance.
     *
     * The frame size is computed automatically based on querying the OS file system for the
     * fundamental block size of the formatted disk.
     *
     * @param hostMemorySize The number of bytes to allocate in host memory for the buffer pool
     */
    BufferMgr(mapd_size_t hostMemorySize);

	/**
	 * @brief A constructor that instantiates a buffer manager instance.
     *
     * The frame size is passed to the constructor as a parameter, along with the host memory size.
     *
     * @param hostMemorySize The number of bytes to allocate in host memory for the buffer pool.
     * @param frameSize The size in bytes of each frame.
	 */
	BufferMgr(mapd_size_t hostMemorySize, mapd_size_t frameSize);

	/**
	 * A destructor that cleans up resources used by a buffer manager instance.
	 */
	~BufferMgr();

    // ***** CHUNK INTERFACE *****/
    
    /**
     * This method returns a reference to a PageInfo, which contains the bounds
     * (address of first and last frames) of the Chunk. The page is automatically
     * pinned, signaling that the page holding the chunk is not to be evicted.
     * The client is then able to perform in-memory operations on the contents of
     * the chunk within its page.
     *
     * @param key
     * @param addr
     * @return
     *
     * @see BufferMgr.cpp for the buffer allocation and eviction/replacement strategies.
     */
    mapd_err_t getChunk(const ChunkKey &key, void *addr);
    

private:
    void *hostMem_;                     /**< A pointer to the host-allocated buffer pool. */
    std::vector<Frame> frames_;         /**< A vector of frames, which compose the buffer pool. */
    const mapd_size_t numFrames;        /**< The number of frames covering the buffer pool space. */
    const mapd_size_t frameSize;        /**< The size of a frame in bytes. */
    
    // void *deviceMem;                 /**< @todo device (GPU) buffer pool */
    
}; // BufferMgr




