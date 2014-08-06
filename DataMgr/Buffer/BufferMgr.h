/**
 * @file	BufferMgr.h
 * @author	Steven Stewart <steve@map-d.com>
 *
 * This file includes the class specification for the buffer manager (BufferMgr), and related
 * data structures and types.
 */
#ifndef DATAMGR_BUFFER_BUFFERMGR_H
#define DATAMGR_BUFFER_BUFFERMGR_H

#include <utility>
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
 * @brief Maps ChunkKey (keys) to Buffer object pointers.
 *
 * The ChunkKeyToBufferMap associates a key, which uniquely identifies a Chunk,
 * with a Buffer in the buffer manager's pool. The existence of such a mapping
 * implies that the Chunk is cached in the buffer pool. The absence of such a
 * mapping implies that the Chunk is not cached, but not necessarily that the
 * Chunk does not exist --  in other words, the Chunk may exist on physical 
 * disk, in which case the file manager should be consulted.
 */
typedef std::map<ChunkKey, Buffer*> ChunkKeyToBufferMap;

/**
 * @class BufferMgr
 * @brief The buffer manager manages a buffer pool consisting of buffers.
 *
 * The buffer manager's job is to manage memory. Upon instantitation, host memory
 * is allocated. The API consists of Buffer-specific and Chunk-specific methods.
 *
 * In essence, the buffer manager serves as an intermediary between the client
 * and the file system. That said, it is possible to create buffers whose contents
 * do not correspond with file system data. Generally speaking, though, the buffer
 * manager will allocate Buffer objects in order to hold/cache the data contained
 * within Chunks, which are uniquely identified using a ChunkKey.
 *
 * @see Buffer.h
 */
class BufferMgr {

public:
    /**
     * @brief The constructor initializes host memory and takes a file manager pointer.
     *
     * @param hostMemSize   The amount of memory to allocate for the host buffer pool.
     * @param fm            A pointer to the file manager resource.
     */
    BufferMgr(mapd_size_t hostMemSize, FileMgr *fm);

    /// Destructor
    ~BufferMgr();

    /**
     * @brief Creates a buffer of the requested pages; returns NULL upon failure.
     *
     * The createBuffer method will search for numPages*pageSize bytes of memory in the
     * free space of the buffer pool. If found, it allocates that memory for a new Buffer
     * object, returning a pointer to said Buffer; otherwise, NULL is returned.
     */
    Buffer* createBuffer(mapd_size_t numPages, mapd_size_t pageSize);

    /// Deletes a buffer
    void deleteBuffer(Buffer *b);

    /**
     * @brief Creates a new chunk (if it doesn't exist). Returns a Buffer object for it.
     *
     * If the Chunk already exists (has an entry in chunkIndex_), then it's Buffer is
     * returned; otherwise, a new Buffer of the requested size is created, and a new
     * entry is inserted into the chunkIndex_ for the specified key. Note that this
     * method does not actually read the Chunk from the file manager into the Buffer.
     * For that functionality, use getChunkBuffer.
     *
     * Also note that, if the Chunk already exists, the numPages and pageSize values
     * have no effect; in other words, the number of pages and the size of the pages
     * will be that of the existing Chunk, which may be different than the arguments
     * passed to this method. A warning is printed to stderr when this happens.
     *
     * @see getChunkBuffer
     *
     * @param key       The unique identifier for the Chunk.
     * @param numPages  The number of pages in the buffer to allocate for the Chunk.
     * @param pageSize  The size of each page allocated for the Chunk.
     * @return          A pointer to the Buffer object intended to hold the Chunk, or NULL
     */
    Buffer* createChunk(const ChunkKey &key, mapd_size_t numPages, mapd_size_t pageSize);

    /**
     * @brief Deletes a chunk: removes it from both the buffer pool (if needed) and file manager.
     *
     * This method will remove a chunk from the buffer pool by freeing the pages belonging to
     * the Buffer object holding the chunk, and then calling the file manager to delete the chunk
     * from the file system by calling file manager's corresponding deleteChunk() method.
     *
     * Note that this method will delete the chunk, assuming the chunk exists, from the file system
     * regardless of whether or not the chunk is currently in the buffer pool.
     *
     * This method should be used cautiously and should NOT be confused with simply evicting
     * the buffer from the buffer pool without removing the chunk from the file system.
     *
     * This method returns two bool values in a pair. The first indicates whether or not the 
     * chunk was removed from the buffer pool, which will be always be true in the case that the
     * chunk wasn't in the buffer pool to begin with. The second bool indicates whether or not
     * the file manager successfully removed the chunk from the file system.
     *
     * @param key                       The unique identifier for a Chunk.
     * @return std::pair<bool,bool>     Existence of chunk in buffer pool and file system.
     */
    std::pair<bool, bool> deleteChunk(const ChunkKey &key);

    /**
     * @brief Returns a Buffer containing the desired chunk, or NULL on error
     *
     * This method will first check if the Chunk is already cached (in the buffer pool),
     * in which case it will return a pointer to the Buffer holding the Chunk. Next, it
     * will attempt to create a Buffer, consisting of the requested pages, and then
     * request the Chunk contents from the file manager.
     *
     * @param key   The unique identifier of the Chunk.
     * @return      A pointer to the Buffer object containing the Chunk, or NULL
     */
    Buffer* getChunkBuffer(const ChunkKey &key);

    /**
     * @brief If cached, returns pointer to cached Chunk and optionally returns the length.
     *
     * This method will return the starting memory address of the buffer holding the
     * Chunk's contents in the host buffer pool, and it will optionally return the
     * length (number of used bytes) of the Chunk via a pointer argument called
     * "length."
     *
     * Note that NULL is returned in the case that the Buffer for the Chunk is
     * currently pinned.
     *
     * See the following sample usage:
     *
     *      mapd_size_t chunkLength;
     *      mapd_addr_t chunkAddr = getChunkAddr(key, &chunkLength);
     *      printf("Chunk address = %p\n", chunkAddr);
     *      printf("Chunk length  = %lu\n", chunkLength);
     *
     * @param key       The unqiue identifer of a Chunk.
     * @param length    Optionally returns the number of used bytes of the Chunk.
     */
    mapd_addr_t getChunkAddr(const ChunkKey &key, mapd_size_t *length = NULL);

    /// Returns whether or not the chunk is cached (in the buffer pool)
    bool chunkIsCached(const ChunkKey &key);

    /// If unpinned, flushes the chunk to disk
    bool flushChunk(const ChunkKey &key);

    /// Calls flushChunk on all indexed chunk keys.
    inline void flushAllChunks() {
        for (auto it = chunkIndex_.begin(); it != chunkIndex_.end(); ++it)
            flushChunk(it->first);
    }

    /// Returns the number of unused bytes in the host buffer pool
    inline mapd_size_t unused() {
        mapd_size_t unused = 0;
        for (auto it = freeMem_.begin(); it != freeMem_.end(); ++it)
            unused += it->first;
        return unused;
    }

    /// Prints to stdout a summary of current host memory allocation
    void printMemAlloc();

    /// Print a representation of chunkIndex_ to stdout
    void printChunkIndex();

private:
    FileMgr *fm_;               /// pointer to the file manager
    mapd_size_t hostMemSize_;   /// number of bytes allocated for host buffer pool
    mapd_addr_t hostMem_;       /// beginning memory address of host buffer pool
    
    /// Maps sizes of free memory areas to host buffer pool memory addresses
    std::multimap<mapd_size_t, mapd_addr_t> freeMem_;

    std::list<Buffer*> buffers_;        // a list of the buffers being managed
    ChunkKeyToBufferMap chunkIndex_;    // an index of chunk keys mapped to buffers
    
    /// Looks up a Chunk's buffer in the chunkIndex
    Buffer* findChunkBuffer(const ChunkKey &key);

}; // BufferMgr

} // Buffer_Namespace

#endif // DATAMGR_BUFFER_BUFFERMGR_H

