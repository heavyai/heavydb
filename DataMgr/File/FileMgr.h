/**
 * @file    FileMgr.h
 * @author  Todd Mostak <todd@map-d.com>
 * @author  Steven Stewart <steve@map-d.com>
 * @brief   This file contains the class specification and related data structures for FileMgr.
 *
 * This file contains the FileMgr class specification, and the specifications of several related
 * types and data structures for use by the Map-D file management system.
 *
 * @see File
 */
#ifndef FILEMGR_H
#define FILEMGR_H

#include <vector>
#include <map>
#include <list>
#include <string>

#include "File.h"
#include "../../Shared/types.h"
#include "../../Shared/errors.h"

/**
 * @type ChunkIndexT
 * @brief An index containing all block addresses associated with a chunk.
 *
 * In the indexed allocation approach, each chunk has an index of block addresses
 * associated with it. The index is implemented, here, as a vector.
 */
typedef std::vector<mapd_size_t> ChunkIndexT;

/**
 * @type ChunkToIndexT
 * @brief A map from chunk keys to chunk indices (which contain block addresses).
 *
 * A chunk key (ChunkKey) uniquely identifies a chunk in a file, and a chunk
 * index provides a list of block addresses associated with that chunk. Thus,
 * ChunkToIndexT is a mapping from chunk keys to chunk indices.
 */
typedef std::map<ChunkKey, ChunkIndexT> ChunkToIndexT;

/**
 * @type _ChunkFileT, ChunkFileT, *ChunkFileTP
 * @brief A chunk file type has a File object and a ChunkIndexT.
 *
 * A chunk file is implemented as a file with a chunk index; hence, the chunk
 * file type (ChunkFileT) encapsulates both of these. As well, free blocks
 * in a file must tracked, and this is accomplished using the freeBlocks list.
 */
typedef struct _ChunkFileT {
    File f;
    ChunkToIndexT index;
    std::list<mapd_size_t> freeBlocks;
    mapd_size_t numAllocated;
    mapd_size_t numFree;
} ChunkFileT, *ChunkFileTP;

/**
 * @class FileMgr
 * @brief The file manager manages interactions between the DBMS and the file system.
 *
 * The main job of the file manager is to translate logical block addresses to physical
 * block addresses.  It manages a list of (files_), which are actually containers of
 * chunks, and indexed allocation is used for mapping chunks to block addresses.
 *
 * The file manager must also manager free space. It accomplishes this using a linked
 * list of free block addresses.
 *
 * FileMgr provides a Chunk-level API, and also a block-level API for finer granularity. Care
 * must be taken when using the block-level API such as not to invalidate the indices that map
 * chunk keys to block addresses.
 *
 * @todo buffered writes, asynch vs. synch
 *
 */
class FileMgr {

public:
    FileMgr(const std::string &basePath);
    ~FileMgr();
    
    // ***** CHUNK INTERFACE *****
    
    /**
     *
     *
     * @param fileId
     * @param key
     * @param index
     * @return
     */
    mapd_err_t getChunkIndex(const int fileId, const ChunkKey &key, ChunkIndexT &index) const;

   /**
    * @param fileId
    * @param key
    * @param buf
    * @return
    */
    mapd_err_t getChunk(const int fileId, const ChunkKey &key, void *buf) const;
    
   /**
    *
    * @param fileId
    * @param key
    * @param nblocks
    * @return
    */
    mapd_err_t getChunkSize(const int fileId, const ChunkKey &key, int *nblocks) const;
    
   /**
    *
    * @param fileId
    * @param key
    * @param size
    * @return
    */
    mapd_err_t getChunkSize(const int fileId, const ChunkKey &key, mapd_size_t *size) const;
    
   /**
    *
    * @param fileId
    * @param key
    * @param nblocks
    * @param size
    * @return
    */
    mapd_err_t getChunkSize(const int fileId, const ChunkKey &key, int *nblocks, mapd_size_t *size) const;
    
   /**
    *
    * @param fileId
    * @param key
    * @param requested
    * @param nblocks
    * @return
    */
    mapd_err_t createChunk(const int fileId, ChunkKey &key, const int requested, int *nblocks);
    
   /**
    *
    * @param fileId
    * @param key
    * @return
    */
    mapd_err_t deleteChunk(const int fileId, const ChunkKey &key);
    
   /**
    *
    * @param fileId
    * @param keys
    * @return
    */
    mapd_err_t deleteChunk(const int fileId, const std::vector<ChunkKey> &keys);
    
    
    // ***** BLOCK INTERFACE *****
    
   /**
    *
    * @param fileId
    * @param blockNum
    * @param buf
    * @return
    */
    mapd_err_t getBlock(const int fileId, const mapd_size_t blockNum, void *buf) const;
    
   /**
    *
    * @param fileId
    * @param blockNums
    * @param buf
    * @return
    */
    mapd_err_t getBlock(const int fileId, const std::vector<mapd_size_t> blockNums, void *buf) const;
    
   /**
    *
    * @param fileId
    * @param blockNum
    * @return
    */
    mapd_err_t createBlock(const int fileId, mapd_size_t *blockNum);
    
   /**
    *
    * @param fileId
    * @param blockNum
    * @return
    */
    mapd_err_t deleteBlock(const int fileId, mapd_size_t blockNum);
    
   /**
    *
    * @param fileId
    * @param index
    * @return
    */
    mapd_err_t deleteBlock(const int fileId, const ChunkIndexT &index);
    
    // ***** FILE INTERFACE *****

    /**
     * @brief Adds a file to the file manager repository.
     *
     *
     * @param fileName The name given to the file in physical storage.
     * @param blockSize The logical block size for the blocks in the file.
     * @param numBlocks The number of logical blocks to initially allocate for the file.
     * @param fileId Returns the unique file identifier of the newly added file.
     * @return mapd_err_t
     */
    mapd_err_t addFile(const std::string &fileName, const mapd_size_t blockSize, const mapd_size_t numBlocks, int *fileId);

    /**
     * @brief Deletes a file from the file manager's repository.
     *
     * This method will delete the specified file (files_[fileId]), and free up
     * its related resources.
     *
     * fileId The unique file identifier of the file to be deleted.
     */
    mapd_err_t deleteFile(const int fileId);

    /**
     * @brief Prints a representation of FileMgr's state to stdout
     */
    void print();

private:
    std::string basePath_;              /**< The OS file system path containing the files. */
    std::vector<ChunkFileT> files_;     /**< The vector of files of chunks being managed. */
};

#endif // FILEMGR_H
