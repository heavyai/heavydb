/**
 * @file	BufferMgr.h
 * @author	Steven Stewart <steve@map-d.com>
 *
 * This file includes the class specification for the buffer manager (BufferMgr), and related
 * data structures and types.
 */
#ifndef _BUFFER_MGR_H_
#define _BUFFER_MGR_H_

#include <map>
#include <list>
#include "Buffer.h"
#include "../../Shared/types.h"

namespace File_Namespace {
    class FileMgr;
}
using File_Namespace::FileMgr;

namespace Buffer_Namespace {

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
    /// Constructor
    BufferMgr(mapd_size_t hostMemSize, FileMgr *fm);

    /// Destructor
    ~BufferMgr();

    /// Creates a buffer of numPages*pageSize bytes (with pages)
    Buffer* createBuffer(mapd_size_t numPages, mapd_size_t pageSize);

    /// Deletes a buffer
    void deleteBuffer(Buffer *b);

    /// Creates a new chunk (if it doesn't exist) and returns a Buffer object for it
    Buffer* createChunk(const ChunkKey &key, mapd_size_t numPages, mapd_size_t pageSize);

    /// @brief Returns a buffer containing the desired chunk, or NULL on error
    Buffer* getChunkBuffer(const ChunkKey &key);

    /// If cached, returns pointer to cached Chunk and optionally sets length
    mapd_addr_t getChunkAddr(const ChunkKey &key, mapd_size_t *length = NULL);

    /// Returns whether or not the chunk is cached (in the buffer pool)
    bool chunkIsCached(const ChunkKey &key);

    /// Flush unpinned chunk to disk
    bool flushChunk(const ChunkKey &key);

    /// Calls flushChunk on all indexed chunk keys.
    inline void flushAllChunks() {
        for (auto it = chunkIndex_.begin(); it != chunkIndex_.end(); ++it)
            flushChunk(it->first);
    }

    /// Prints to stdout a summary of current host memory allocation
    void printMemAlloc();

    /// Print a representation of chunkIndex_ to stdout
    void printChunkIndex();

private:
    FileMgr *fm_;
    mapd_size_t hostMemSize_;
    mapd_addr_t hostMem_;
    std::multimap<mapd_size_t, mapd_addr_t> freeMem_;

    std::list<Buffer*> buffers_;
    ChunkKeyToBufferMap chunkIndex_;
    
    /// Looks up a Chunk's buffer in the chunkIndex
    Buffer* findChunkBuffer(const ChunkKey &key);

}; // BufferMgr

} // Buffer_Namespace

#endif // _BUFFER_MGR_H_

