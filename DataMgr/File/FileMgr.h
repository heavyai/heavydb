/**
 * @file    FileMgr.h
 * @author  Todd Mostak <todd@map-d.com>
 * @author  Steven Stewart <steve@map-d.com>
 * @brief   This file contains the class specification and related data structures for FileMgr.
 *
 * This file contains the FileMgr class specification, and the specifications of several related
 * types and data structures for use by the Map-D file management system.
 *
 * It is recommended that any reader of this file first read through the descriptions of all the 
 * types and data structures used by the file manager.
 *
 * The file manager manages files, which are a collection of logical blocks. Each file has its own
 * logical block size (@see FileInfo).
 *
 * @see File.h
 */
#ifndef FILEMGR_H
#define FILEMGR_H

#include <vector>
#include <map>
#include <list>
#include <utility>
#include <string>
#include <cassert>

#include "File.h"
#include "../../Shared/types.h"
#include "../../Shared/errors.h"

/**
 * @type FileInfo
 * @brief A FileInfo type has a file pointer and metadata about a file.
 *
 * A file info structure wraps around a file pointer in order to contain additional
 * information/metadata about the file that is pertinent to the file manager.
 *
 * The free blocks within a file must be tracked, and this is implemented using a
 * linked list. Helper functions are provided: size(), available(), and used().
 */
struct FileInfo {
    int fileId;
    FILE *f;
    std::list<mapd_size_t> freeBlocks;
    mapd_size_t blockSize;
    mapd_size_t nblocks;
    
    inline mapd_size_t size() { return blockSize * nblocks; }
    inline mapd_size_t available() { return freeBlocks.size() * blockSize; }
    inline mapd_size_t used() { return size() - available(); }

    /// Constructor
    FileInfo(int fileId, FILE *f, mapd_size_t blockSize, mapd_size_t nblocks)
         : fileId(fileId), f(f), blockSize(blockSize), nblocks(nblocks)
    {
        assert(f);
        
        // initialize free block list
        for (int i = 0; i < nblocks; ++i)
            freeBlocks.push_back(i);
    }
    
    /// Destructor
    ~FileInfo() {
        mapd_err_t err = File::close(f);
        if (err != MAPD_SUCCESS)
            fprintf(stderr, "[%s:%d] Error closing file %d.\n", __func__, __LINE__, fileId);
    }
};

/**
 * @type FileMap
 * @brief Maps file identifiers to file pointers.
 *
 * This type maps file identifiers (int fileId) to FileInfo objects.
 */
typedef std::map<int, FileInfo*> FileMap;

/**
 * @type BlockAddr
 * @brief A block address that uniquely identifies a block within a file.
 *
 * A block address type includes a file identifier and a block address. The block
 * address (or block number) uniquely identifies a block within the respective file.
 */
struct BlockAddr {
	int fileId;
	mapd_size_t blockAddr;
};

/**
 * @brief This struct wraps a block address with metadata associated with that block.
 *
 * This struct provides metadata about blocks. It has the address of the block within its
 * respective file via BlockAddr, and it has metadata about the block's size, the address
 * of the last block written to the block, and the epoch. The epoch is temporal reference
 * marker indicating the last time that the block was updated.
 *
 * Note that a block may be flagged as a "shadow" to indicate that a more up-to-date version
 * of the block exists. This flag is typically set by the buffer manager, which may have read
 * the block into an in-memory copy called a page, and which may have been "dirtied" (updated).
 */
struct BlockInfo {
	BlockAddr blk;

	// metadata about a logical block
	mapd_size_t blockSize;			/**< the logical block size in bytes */
	mapd_size_t endByteOffset;		/**< the last address of the last byte written to the block */
	unsigned int epoch;				/**< indicates the last temporal reference point for which the block is current */
	bool isShadow;					/**< indicates whether or not the block is a shadow copy */
    bool isEmpty;                   /**< indicates that the block does not contain any real data (is empty) */
};

/**
 * @type Chunk
 * @brief A Chunk is the fundamental unit of execution in Map-D.
 *
 * A chunk is composed of blocks. These blocks can exist across multiple files managed by
 * the file manager. The Chunk type is a vector of BlockInfoT, which is a struct containing
 * block information (file, address, and metadata).
 */
typedef std::vector<BlockInfo> Chunk;

/**
 * @type ChunkKeyToChunkMap
 * @brief A map from chunk keys to files to block addresses.
 *
 * The file system can store multiple chunks across multiple files. With that
 * in mind, the challenge is to be able to reconstruct the blocks that compose
 * a chunk upon request. A chunk key (ChunkKey) uniquely identifies a chunk,
 * and so ChunkKeyToChunkMap maps chunk keys to the blocks that compose it.
 *
 * A ChunkKey is mapped to a vector of BlockInfoT, the latter of which represents
 * the in-order listing of blocks for the respective chunk. The file manager will
 * bring these blocks into contiguous order in memory upon request, and will do so
 * by looking up the ChunkKey via this mapping. BlockInfoT has a BlockAddr and
 * metadata about the block.
 *
 * BlockAddr is simply a pair of values: the file identifier, and the block
 * address within that file. (Note that although files are heterogeneous, logical
 * blocks are homogeneous; namely, that data from at most one chunk can populate
 * a block.)
 */
typedef std::map<ChunkKey, Chunk> ChunkKeyToChunkMap;

/**
 * @class FileMgr
 * @brief The file manager manages interactions between the DBMS and the file system.
 *
 * The main job of the file manager is to translate logical block addresses to physical
 * block addresses.  It manages a list of FileInfo objects, and files are actually containers
 * of chunks. Indexed allocation is used for mapping chunks to block addresses.
 *
 * The file manager must also manage free space. It accomplishes this using a linked list of
 * free block addresses associated with each file via struct FileInfo.
 *
 * FileMgr provides a chunk-level API,  a block-level API for finer granularity, and a File-level
 * API. Care must be taken when using the block-level API such as not to invalidate the indices
 * that map chunk keys to block addresses.
 *
 * @todo asynch vs. synch?
 *
 */
class FileMgr {

public:
    static unsigned nextFileId;
    
    FileMgr(const std::string &basePath);
    ~FileMgr();
    
   // ***** FILE INTERFACE *****

   /**
    * @brief Adds a file to the file manager repository.
    *
    * This method will create a FileInfo object for the file being added, and it will create
    * the corresponding file on physical disk with the indicated number of blocks pre-allocated.
    *
    * A pointer to the FileInfo object is returned, which itself has a file pointer (FILE*) and
    * a file identifier (int fileId).
    *
    * @param fileName The name given to the file in physical storage.
    * @param blockSize The logical block size for the blocks in the file.
    * @param numBlocks The number of logical blocks to initially allocate for the file.
    * @param err Holds an error code, should an error occur.
    * @return FileInfo* A pointer to the FileInfo object of the added file.
    */
   FileInfo* createFile(const mapd_size_t blockSize, const mapd_size_t nblocks, mapd_err_t *err);

   /**
    * @brief Deletes a file from the file manager's repository.
    *
    * This method will delete the specified file from the FileMap (files_). Note that it does
    * not remove references to the fileId of the deleted file. Any method that uses the 
    * fileId will need to check for the existence of the file in the FileMap.
    *
    * @param fileId The unique file identifier of the file to be deleted.
    * @return An error code, should an error occur.
    */
    mapd_err_t deleteFile(const int fileId);

   /**
    * @brief Finds the file in the file manager's FileMap (files_).
    *
    * @param fileId The unique file identifier of the file to be found.
    * @param err An error code, should an error occur.
    * @return A pointer to the found FileInfo object for the file.
    */
    FileInfo* findFile(const int fileId, mapd_err_t *err);
    
    // ***** CHUNK INTERFACE *****
    
    /**
     *
     *
     * @param fileId
     * @param key
     * @param index
     * @return
     */
    //Chunk* getChunkRef(const ChunkKey &key, mapd_err_t *err);

   /**
    * @param fileId
    * @param key
    * @param buf
    * @return
    */
    //Chunk* getChunkCopy(const ChunkKey &key, void *buf, mapd_err_t *err) ;
    
   /**
    * This method returns the number of blocks that composes the chunk identified
    * by the key, and the size in bytes occupied by those blocks. Since blocks may
    * be scattered across multiple files, it's possible for blocks to differ in size.
    * This method accounts for this possibility when computing the size.
    *
    * @param key The unique identifier of a Chunk.
    * @param nblocks A return value that will hold the number of blocks of the chunk.
    * @param size A return value that will hold the size in bytes occupied by the blocks of the chunk.
    * @return MAPD_FAILURE or MAPD_SUCCESS
    */
    //mapd_err_t getChunkSize(const ChunkKey &key, int *nblocks, mapd_size_t *size) const;

   /**
    * This method returns the actual number of bytes occupied by a chunk. This calculation
    * of the size of the chunk adds up the number of actual bytes occupied in each block as
    * opposed to adding up the block sizes. This is necessary to account for partially
    * filled blocks.
    *
    * @param key The unique identifier of a Chunk.
    * @param size A return value that will hold the actual size in bytes occupied by the blocks of the chunk.
    * @return MAPD_FAILURE or MAPD_SUCCESS
    */
    //mapd_err_t getChunkActualSize(const ChunkKey &key, mapd_size_t *size) const;

   /**
    * Given a key, this method requests the file manager to create a new chunk of the requested
    * number of bytes (size). A pointer to the new Chunk object is returned, or NULL upon failure.
    * If failure occurs, an error code may be stored in err.
    *
    * If src is NULL, then each block of the chunk will have its "isNull" flag set to true; otherwise,
    * the data in src will be written to the new chunk.
    *
    * @param key The unique identifier of the new chunk.
    * @param size The amount of memory requested for the new chunk.
    * @param blockSize The size of the logical disk blocks for which the chunk will be stored
    * @param src The source data to be copied into the chunk (can be NULL).
    * @param err An error code, should an error happen to occur.
    * @return A pointer to a new Chunk, or NULL.
    */
    //Chunk* createChunk(ChunkKey &key, const mapd_size_t size, const mapd_size_t blockSize, const void *src, mapd_err_t *err);
    
   /**
    * Given a chunk key, this method deletes a chunk from the file system. It returns the number of
    * blocks freed and the amount of memory freed.
    *
    * @param key The unique identifier of the chunk.
    * @param nblocks The number of blocks freed (can be NULL).
    * @param size The number of bytes freed (can be NULL).
    * @return MAPD_FAILURE or MAPD_SUCCESS
    */
    //mapd_err_t deleteChunk(const ChunkKey &key, mapd_size_t *nblocks, mapd_size_t *size);
    
    // ***** BLOCK INTERFACE *****
    
   /**
    *
    * @param fileId
    * @param blockNum
    * @param buf
    * @return
    */
    //BlockAddr* getBlock(void *buf, mapd_err_t *err) const;
    
   /**
    *
    * @param fileId
    * @param blockNums
    * @param buf
    * @return
    */
    //BlockAddr* getBlock(const int fileId, void *buf, mapd_err_t *err) const;
    
   /**
    *
    * @param fileId
    * @param blockNum
    * @return
    */
    //BlockAddr* createBlock(const int fileId, mapd_err_t *err);
    
   /**
    *
    * @param fileId
    * @param index
    * @return
    */
    //mapd_err_t deleteBlock(const int fileId, const BlockAddr &index);
    
    /**
     * @brief Prints a representation of FileMgr's state to stdout
     */
    //void print();

private:
    std::string basePath_;          /**< The OS file system path containing the files. */
    FileMap files_;                 /**< Maps file identifiers (int) to FileInfo objects. */
    ChunkKeyToChunkMap chunkIndex_; /**< Index for looking up chunks, which are vectors of BlockAddr */
};

#endif // FILEMGR_H
