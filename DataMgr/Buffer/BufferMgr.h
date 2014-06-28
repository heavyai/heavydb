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
 * The interface to BufferMgr includes the most basic functionality: the ability to bring disk pages into 
 * the buffer pool, to pin it, and to unpin it. A chunk-level and frame-level API are provides, offering
 * two levels of interaction with the buffer pool in terms of granularity.
 *
 * Frames are allocated upon allocating host memory. There's one frame per frameSize bytes, and
 * they are ordered from 0 to (numFrames-1).
 *
 * Pages are variabled-sized collection of frames. They can occupy any contiguous subset of frames,
 * and are not necessarily ordered because their allocation depends on access to free memory, as
 * well as the replacement algorithm being used.
 */
class BufferMgr {

public:
    /**
     * @brief A constructor that instantiates a buffer manager instance, and allocates the buffer pool.
     *
     * The constructor allocates the buffer pool in host memory.
     *
     * The frame size is computed automatically based on querying the OS file system for the
     * fundamental block size of the formatted disk.
     *
     * @param hostMemorySize The number of bytes to allocate in host memory for the buffer pool
     */
    BufferMgr(mapd_size_t hostMemorySize);

	/**
	 * @brief A constructor that instantiates a buffer manager instance, and allocates the buffer pool.
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

    /**
     * @brief Gets the requested chunk by returning a pointer to its PageInfo object.
     *
     * This method will check if the chunk is already in the buffer pool. If it is,
     * then the pin count for its PageInfo struct is incremented. If the chunk is not
     * in the buffer pool, then the buffer manager needs to find a set of frames to
     * bring the chunk into, wrapping this up into a new PageInfo struct, and updating
     * ChunkToPageMap accordingly.
     *
     * If it is necessary to replace an existing PageInfo, then its dirty frames will be flushed
     * before bringing in the new chunk.
     *
     * If for some reason it is not possible to bring the chunk into the buffer pool, then an error
     * code is returned (MAPD_ERR_BUFFER). Upon success, a pointer to the new PageInfo struct is
     * returned in page; otherwise, it is NULL.
     *
     * @param key
     * @param addr
     * @return
     *
     * @see BufferMgr.cpp for the buffer allocation and eviction/replacement strategies.
     */
    mapd_err_t getChunkHost(const ChunkKey &key, PageInfo *page);
    
    /**
     * @brief Returns true if the chunk is cached in the host buffer pool.
     * @param key The chunk key uniquely identifies the chunk.
     */
    bool isCachedHost(const ChunkKey &key);
    
    /**
     * @brief Returns a float representing the host hit rate.
     * @return float The hit rate for the host.
     */
    inline float hostHitRate() {
        return float(numHitsHost_) / float(numHitsHost_ + numMissHost_);
    }
    
    /**
     * @brief Sets the hit and miss counts for the host back to 0.
     */
    inline void resetHitRateHost(){
        numHitsHost_ = 0;
        numMissHost_ = 0;
    }
    
    /**
     * @brief Prints a summary of the frames (Frame objects) in the host buffer pool to stdout.
     *
     * Traverses the vector frames_ in order to print a summary of the chunks currently cached in
     * the host buffer pool.
     */
    void printFramesHost();
    
   /**
    * @brief Prints a summary of the pages (Page objects) in the host buffer pool to stdout.
    *
    * Traverses the vector pages_ in order to print a summary of the pages currently cached in
    * the host buffer pool.
    */
   void printPagesHost();
    
    /**
     * @brief Prints a summary of the chunks in the host buffer pool to stdout.
     *
     * Traverses ChunkToPageMap in order to print a summary of the chunks currently cached in
     * the host buffer pool.
     */
    void printChunksHost();

private:
    void *hostMem_;                     /**< A pointer to the host-allocated buffer pool. */

    // Frames
    std::vector<Frame> frames_;         /**< A vector of in-order frames, which compose the buffer pool. */
    std::list<Frame*> free_;            /**< A linked list of pointers to free frames. */
    const mapd_size_t numFrames;        /**< The number of frames covering the buffer pool space. */
    const mapd_size_t frameSize;        /**< The size of a frame in bytes. */

    // Pages
    std::vector<PageInfo> pages_;       /**< A vector of pages, which are present in the buffer pool. */
    const mapd_size_t numPages;         /**< The number of pages currently in the buffer pool. */
    
    // void *deviceMem;                 /**< @todo device (GPU) buffer pool */
    
    // Metadata
    unsigned numHitsHost_;              /**< The number of host memory cache hits. */
    unsigned numMissHost_;              /**< The number of host memory cache misses. */
    
}; // BufferMgr




