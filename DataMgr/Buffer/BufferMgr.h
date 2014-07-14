/**
 * @file	BufferMgr.h
 * @author	Steven Stewart <steve@map-d.com>
 *
 * This file includes the class specification for the buffer manager (BufferMgr), and related
 * data structures and types.
 */
#include <vector>
#include <string>
#include <cassert>
#include <map>
#include <list>
#include <pair>
#include "../../Shared/types.h"
#include "../../Shared/errors.h"

namespace Buffer_Namespace {

// Forward declaration(s)
class FileMgr;

/**
 * @type Frame
 * @brief A frame has an address and a flag that indicates if it is dirty.
 */
struct Frame {
    mapd_size_t addr;
    bool isDirty;
};

/**
 * @brief PageBounds provides support for various page sizes.
 *
 * The PageBounds refers to a span of consecutive frames. If the bounds are (i, j),
 * then the frames are those where i < j.
 */
typedef std::pair <mapd_size_t, mapd_size_t> PageBounds;

/**
 * A PageInfo struct contains the bounds of the page (beginning and ending
 * frame IDs) and metadata about the page. Although a page's properties are
 * publicly exposed, helper methods are provided for convenience.
 */
struct PageInfo {
    PageBounds bounds;  /**< the span of frame IDs occupied by this page */
    int pinCount;       /**< the number of pins (resources using the page) */
    bool dirty;         /**< indicates the page has been altered and differs from disk */
    bool inHostPool;    /**< indicates that the page is currently in the host pool */

    PageInfo() {
        pinCount = 0;
        dirty = false;
        inHostPool = false;
        bounds.first = 0;
        bounds.second = 0;
    }

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
 * @brief A ChunkKeyToPageMap maps a chunk key to a page in the buffer pool.
 *
 * If a chunk currently exists in the buffer pool, then the ChunkKeyToPageMap maps the chunk, 
 * via its key, to its page in the buffer pool.
 */
typedef std::map<ChunkKey, PageInfo*> ChunkKeyToPageMap;

/**
 * @class 	BufferMgr
 * @author	Steven Stewart <steve@map-d.com>
 * @brief The buffer manager handles the caching and movement of data within the memory hierarchy (CPU/GPU).
 *
 * The buffer manager is the subsystem responsible for the allocation of the buffer space (also called the
 * memory pool or memory cache). 
 *
 * The buffer manager maintains a cache of pages in memory to hold recently accessed data. In
 * general, it acts as an intermediary between DBMS modules and the file manager. The main
 * goal of the buffer manager is the minimization of physical I/O.
 *
 * More specifically, the Map-D buffer manager keeps a cache of "chunks" in memory. If the
 * requested chunk is not in the cache, then the buffer manager requests the chunk from
 * the file manager and places it in a memory page of sufficient size. The page, which
 * itself can consist of a variable number of fixed size frames, may be padded with empty
 * space in order to leave room for appending new data.
 *
 * Map-D uses a three-level storage hierarchy: nonvolatile (disk), main memory, and GPU memory. The buffer
 * manager handles the caching and movement of data across this hierarchy. 
 *
 * The interface to BufferMgr includes the most basic functionality: the ability to bring "chunks" into 
 * the buffer pool, to pin it, and to unpin it. A chunk-level and frame-level API are provided, offering
 * two levels of interaction with the buffer pool in terms of granularity.
 *
 * Host memory is allocated upon construction of a BufferMgr object, and there's one frame per frameSize
 * bytes, and they are ordered from 0 to (numFrames-1).
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
    BufferMgr(mapd_size_t hostMemorySize, FileMgr *fm);

	/**
	 * @brief A constructor that instantiates a buffer manager instance, and allocates the buffer pool.
     *
     * The frame size is passed to the constructor as a parameter, along with the host memory size.
     *
     * @param hostMemorySize The number of bytes to allocate in host memory for the buffer pool.
     * @param frameSize The size in bytes of each frame.
	 */
	BufferMgr(mapd_size_t hostMemorySize, mapd_size_t frameSize, FileMgr *fm);

	/**
	 * A destructor that cleans up resources used by a buffer manager instance.
	 */
	~BufferMgr();

    /**
     * @brief Returns a pair of bool: (1) true if chunk is in chunkIndex_; (2) true if chunk is cached
     * @param key The unique identifier for the chunk.
     */
    std::pair<bool> chunkStatus(const ChunkKey &key);
    
   /**
    * @brief 
    * @param 
    */
    bool insertIntoIndex(std::pair<ChunkKey, PageInfo*> e);

    /**
     * @brief Returns a pointer to a PageInfo object for a chunk cached in host memory.
     *
     * This method returns a pointer to a page (PageInfo object) in host memory containing
     * the requested chunk. If "pin" is true, then the page is automatically pinned.
     *
     * If the chunk is not in the buffer pool, then the buffer manager needs to find a set of
     * frames to compose a new PageInfo object, and will update ChunkKeyToPageMap accordingly.
     *
     * If it is necessary to replace an existing PageInfo, then its dirty frames will be flushed
     * before bringing in the new chunk.
     *
     * Upon failure to bring a chunk onto a page, NULL is returned.
     *
     * @param key The unique identifier for the requested chunk.
     * @param pin By default, set to true, which pins the page for the chunk.
     * @return PageInfo* NULL on error; otherwise, a pointer to the chunk's page in host memory.
     */
    PageInfo* getChunk(const ChunkKey &key, bool pin = true);
    
    /**
     * This method does the same thing as getChunkHost(key, pin), but leaves "pad" bytes of
     * extra space at the end of the page holding the chunk. If the chunk is already cached,
     * then its page bounds are extended to accomodate the padding.
     */
    PageInfo* getChunk(const ChunkKey &key, mapd_size_t pad, bool pin = true);

    /**
     * @brief Updates the chunk.
     *
     * @param key       The unique identifier of the chunk.
     * @param offset    The offset address where the update begins.
     * @param size      The number of bytes being written.
     * @param src       A pointer to the source of the data being written.
     */
    void updateChunk(const ChunkKey &key, mapd_size_t offset, mapd_size_t size, mapd_byte_t *src);

    /**
     * @brief Removes the chunk from the buffer pool. Returns false if the chunk is pinned.
     * @param key       The unique identifier of the chunk.
     */
    bool removeChunk(const ChunkKey &key);

    /**
     * @brief Append data to a chunk.
     *
     * @param key       The unique identifier of the chunk.
     * @param size      The size of the source data in bytes.
     * @param src       The source location of the data to be appended.
     */
    void appendChunk(const ChunkKey &key, mapd_size_t size, mapd_byte_t *src);

    /**
     * @brief Flushes all the dirty frames of the specified chunk from the host buffer pool.
     * @param key The unique identifier of the chunk.
     * @param epoch Identifier of the most recent checkpoint.
     */
    void flushChunk(const ChunkKey &key, unsigned int epoch);

    /**
     * @brief Flushs all dirty pages currently in the host buffer pool.
     * @param epoch Identifier of the most recent checkpoint.
     */
    void flushAll(unsigned int epoch);

    /**
     * @brief Flushs all dirty pages currently in the host buffer pool.
     * @param epoch Identifier of the most recent checkpoint.
     */
    void pinChunk(const ChunkKey &key);

    /**
     * @brief Flushs all dirty pages currently in the host buffer pool.
     * @param epoch Identifier of the most recent checkpoint.
     */
    void unpinChunk(const ChunkKey &key);

    /**
     * @brief Returns the number of unpinned frames available.
     */
    inline mapd_size_t availableFrames() { return free_.size(); }

    /**
     * @brief Returns the number of unpinned bytes available.
     */
    inline mapd_size_t available() { return available() * frameSize_; }

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
     * Traverses ChunkKeyToPageMap in order to print a summary of the chunks currently cached in
     * the host buffer pool.
     */
    void printChunksHost();

private:
    BufferMgr(const BufferMgr&);
    BufferMgr& operator=(const BufferMgr&);

    FileMgr *fm_;                       /**< pointer to a file manager object */
    mapd_byte_t *hostMem_;              /**< pointer to the host-allocated buffer pool. */

    // Frames
    std::vector<Frame*> frames_;        /**< A vector of in-order frames, which compose the buffer pool. */
    std::list<Frame*> free_;            /**< A linked list of pointers to free frames. */
    mapd_size_t frameSize_;             /**< The size of a frame in bytes. */

    // Pages
    std::vector<PageInfo*> pages_;      /**< A vector of pages, which are present in the buffer pool. */

    // void *deviceMem;                 /**< @todo device (GPU) buffer pool */
    
    // Data structures
    ChunkKeyToPageMap chunkIndex_;
    
    // Metadata
    unsigned numHitsHost_;              /**< The number of host memory cache hits. */
    unsigned numMissHost_;              /**< The number of host memory cache misses. */
    
}; // BufferMgr

} // Buffer_Namespace


