/**
 * @file	BufferMgr.h
 * @author	Steven Stewart <steve@map-d.com>
 *
 * This file includes the class specification for the buffer manager (BufferMgr), and related
 * data structures and types.
 */
#include <map>
#include <list>
#include "../../Shared/types.h"

namespace File_Namespace {
    class FileMgr;
}
using File_Namespace::FileMgr;

namespace Buffer_Namespace {

/**
 * @struct Page
 */
struct Page {
    mapd_addr_t addr = 0;
    bool dirty = false;

    Page(mapd_size_t addrIn, bool dirtyIn = false)
        : addr(addrIn), dirty(dirtyIn) {}
};

/**
 * @struct Buffer
 */
struct Buffer {
    mapd_addr_t begin = 0;
    mapd_addr_t end = 0;
    int pins = 0;
    bool dirty = false;
    std::vector<Page*> pages;

    /// Returns the size in bytes of the buffer
    inline mapd_size_t size() { return end-begin; }

    /// Increments the pin count
    inline void pin() { pins++; }

    /// Decrements the pin count
    inline void unpin() { pins--; }
};

/**
 * @type ChunkKeyToBufferMap
 */
typedef std::map<ChunkKey, Buffer*> ChunkKeyToBufferMap;

/**
 * @class BufferMgr
 *
 */
class BufferMgr {

public:
    /**
     * Constructor
     *
     * @param numPages
     * @param pageSize
     * @param fm
     */
    BufferMgr(mapd_size_t numPages, mapd_size_t pageSize, FileMgr *fm);

    /// Destructor
    ~BufferMgr();

    /**
     * Creates a buffer containing enough pages to hold n bytes. The buffer
     * is automatically pinned in the buffer pool.
     *
     * @param n         The number of bytes to copy from src into the buffer.
     * @return int      A pointer to the new Buffer object.
     */
    Buffer* createBuffer(mapd_size_t n);

    /**
     * Updates a buffer with the given id, copying n bytes from src to the 
     * specified offset into the buffer. Returns the number of bytes copied.
     * The buffer manager will attempt to extend the buffer or reallocate it
     * if it needs additional memory, and will return 0 in the case that it
     * fails.
     *
     * @param n         The number of bytes to copy from src into the buffer.
     * @param src       The source memory address from which data is copied.
     * @return int      The id of the new buffer in the BufferMgr's index.
     */
    mapd_size_t updateBuffer(Buffer &b, mapd_addr_t offset, mapd_size_t n, mapd_addr_t *src);

    /**
     * Removes a buffer having the specified index.
     * @param id        The id of the buffer in the BufferMgr index.
     */
    void deleteBuffer(Buffer *b);

    /**
     * Copies the first n bytes of the source buffer (id = src_id) to the
     * destination buffer (id = dest_id). Returns the number of bytes copied,
     * or 0 on error.
     *
     * @param src_id       The id of the source buffer.
     * @param dest_id      The id of the destination buffer.
     * @return mapd_size_t The number of bytes copied (0 on error).
     */
    mapd_size_t copyBuffer(Buffer &bSrc, Buffer &bDest, mapd_size_t n);

    /**
     * Concatenates the buffer with id2 to the end of the buffer with id2.
     * If id1 does not have enough empty space, then a new buffer is created
     * of the combined size of the two buffers. Any empty space in either
     * buffer is pushed to the end of the new buffer, and the two original
     * buffers are deleted; thus, the return value is a pointer to the new
     * buffer, whose id may be different than id1.
     *
     * @param id1       The id of the first buffer.
     * @param id2       The id of the second buffer.
     * @return Buffer*  A pointer to the concatenated buffer.
     */
    Buffer* concatBuffer(Buffer &b1, Buffer &b2);

    /**
     * @brief Requests that a chunk be loaded into a buffer from storage.
     *
     * This method requests that a chunk be loaded from storage into a buffer.
     * If the chunk cannot be loaded into a buffer, then false is returned.
     *
     * If the client knows the chunk will fit in the buffer, then setting fast to
     * true will avoid unnecessary checks and result in faster execution.
     *
     * If true is returned, the buffer's pin count is incremented by 1.
     */
    bool loadChunk(const ChunkKey &key, Buffer &b, bool fast = false);

    /**
     * @brief Returns a pointer to a Buffer object containing the requested chunk and pins the buffer.
     *
     * @param ChunkKey  The unique identifier of a chunk.
     * @return Buffer*  Returns the buffer containing the chunk, or NULL if it's not found.
     */
    inline Buffer* getChunkBuffer(const ChunkKey &key) {
        Buffer *b = findChunkPage(key);
        if (b) b->pin();
        return b;
    }

    /**
     * @brief Returns the memory address that points to the beginning of the chunk. Pins the chunk's buffer.
     *
     * @param ChunkKey  The unique identifier of a chunk.
     * @return Buffer*  Returns the buffer containing the chunk, or NULL if it's not found.
     */
    inline mapd_addr_t* getChunkAddr(const ChunkKey &key, mapd_size_t *size = NULL) {
        Buffer *b = findChunkPage(key);
        if (b) {
            b->pin();
            if (size) *size = b->size();
            return hostMem_ + b->begin;
        }
        return NULL;
    }

    /**
     * @brief Flushes the contents of the chunk's buffer to storage.
     *
     * This method flushes the contents of the chunk's buffer to storage. If all
     * is true, then all pages are flushed, regardless of their dirty status;
     * otherwise, only dirty pages are flushed. If force is true, then the 
     * the buffer is flushed even if its dirty flag is false. In general,
     * the client should not need to use "all" or "force.""
     */
    void flushChunk(Buffer &b, bool all = false, bool force = false);

    /// Prints to stdout a summary of current host memory allocation
    void printMemAlloc();

private:
    FileMgr *fm_;
    mapd_addr_t *hostMem_;
    mapd_size_t hostMemSize_;
    mapd_size_t pageSize_;
    mapd_addr_t nextPage_;
    std::list<Buffer*> buffers_;
    ChunkKeyToBufferMap chunkIndex_;

    inline void setDirtyPages(Buffer &b, mapd_addr_t begin, mapd_addr_t end) {
        auto it = b.pages.begin();
        it += begin / pageSize_;
        if (it != b.pages.end())
            while ((*it)->addr < end)
                (*it)->dirty = true;
        b.dirty = true;
    }

    inline Buffer* findChunkPage(const ChunkKey key) {
        auto it = chunkIndex_.find(key);
        if (it == chunkIndex_.end()) // not found
            return NULL;
        return it->second;
    }

}; // BufferMgr

} // Buffer_Namespace


