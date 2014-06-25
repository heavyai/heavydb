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
#include <utility>
#include <string>

#include "File.h"
#include "../../Shared/types.h"
#include "../../Shared/errors.h"

/**
 * @type BlockAddrT
 * @brief A block address that uniquely identifies a block within a file.
 *
 * A block address type includes a file identifier and a block address. The block
 * address (or block number) uniquely identifies a block within the respective file.
 */
typedef struct BlockAddr {
	int fileId;
	mapd_size_t blockAddr;
};

/**
 * @brief This struct wraps a block address with metadata associated with that block.
 *
 * This struct provides metadata about blocks. It has the address of the block within its
 * respective file via BlockAddrT, and it has metadata about the block's size, the address
 * of the last block written to the block, and the epoch. The epoch is temporal reference
 * marker indicating the last time that the block was updated.
 */
typedef struct BlockInfo {
	BlockAddr blk;

	// metadata about a logical block
	mapd_size_t blockSize;			/**< the logical block size in bytes */
	mapd_size_t endByteOffset;		/**< the last address of the last byte written to the block */
	unsigned int epoch;				/**< indicates the last temporal reference point for which the block is current */
	bool isShadow;					/**< */
};

/**
 * @type Chunk
 * @brief A Chunk is the fundamental unit of execution in Map-D.
 *
 * A chunk is composed of a blocks. These blocks can exist across multiple files
 * managed by the file manager. The Chunk type is a vector of BlockInfoT.
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
 * by looking up the ChunkKey via this mapping. BlockInfoT has a BlockAddrT and
 * metadata about the block.
 *
 * BlockAddrT is simply a pair of values: the file identifier, and the block
 * address within that file. (Note that although files are heterogeneous, logical
 * blocks are homogeneous; namely, that data from at most one chunk can populate
 * a block.)
 */
typedef std::map<ChunkKey, Chunk> ChunkKeyToChunkMap;

/**
 * @type FileInfo, FileInfoT, *FileInfoTP
 * @brief A chunk file type has a File object and a BlockAddrT.
 *
 * A file info structure wraps around a File object in order to contain additional
 * information/metadata about the file that is pertinent to the file manager.
 *
 * The free blocks within a file must be tracked, and this is implemented using a
 * linked list.
 */
typedef struct FileInfo {
    File f;
    std::list<mapd_size_t> freeBlocks;

    // FileInfo(mapd_size_t nblocks);
};

/**
 * @class FileMgr
 * @brief The file manager manages interactions between the DBMS and the file system.
 *
 * The main job of the file manager is to translate logical block addresses to physical
 * block addresses.  It manages a list of (files_), which are actually containers of
 * chunks, and indexed allocation is used for mapping chunks to block addresses.
 *
 * The file manager must also manager free space. It accomplishes this using a linked
 * list of free block addresses associated with each file via struct FileInfo.
 *
 * FileMgr provides a chunk-level API, and also a block-level API for finer granularity. Care
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
    mapd_err_t getChunkRef(const ChunkKey &key, Chunk &c) const;

   /**
    * @param fileId
    * @param key
    * @param buf
    * @return
    */
    mapd_err_t getChunkCopy(const ChunkKey &key, void *buf) const;
    
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
    mapd_err_t getChunkSize(const ChunkKey &key, int *nblocks, mapd_size_t *size) const;

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
    mapd_err_t getChunkActualSize(const ChunkKey &key, mapd_size_t *size) const;

   /**
    * Given a key, this method requests the file manager to create a new chunk,
    * requesting the requested in bytes.
    *
    * @param key The unique identifier of the new chunk.
    * @param requested The amount of memory requested for the new chunk.
    * @param actual The amount of memory allocated for the new chunk.
    * @param src The source data to be copied into the chunk (can be NULL).
    * @return MAPD_FAILURE or MAPD_SUCCESS @todo MAPD_ERR_DUP_KEY, DISK_FULL
    */
    mapd_err_t createChunk(ChunkKey &key, const mapd_size_t requested, mapd_size_t *actual, const void *src);
    
   /**
    * Given a chunk key, this method deletes a chunk from the file system. It returns the number of
    * blocks freed and the amount of memory freed.
    *
    * @param key The unique identifier of the chunk.
    * @param nblocks The number of blocks freed (can be NULL).
    * @param size The number of bytes freed (can be NULL).
    * @return MAPD_FAILURE or MAPD_SUCCESS
    */
    mapd_err_t deleteChunk(const ChunkKey &key, mapd_size_t *nblocks, mapd_size_t *size);
    
    // ***** BLOCK INTERFACE *****
    
   /**
    *
    * @param fileId
    * @param blockNum
    * @param buf
    * @return
    */
    mapd_err_t getBlock(const &BlockAddrT blk, void *buf) const;
    
   /**
    *
    * @param fileId
    * @param blockNums
    * @param buf
    * @return
    */
    mapd_err_t getBlock(const int fileId, mapd_size_t blockAddr, void *buf) const;
    
   /**
    *
    * @param fileId
    * @param blockNum
    * @return
    */
    mapd_err_t createBlock(const int fileId, mapd_size_t *blockAddr);
    
   /**
    *
    * @param fileId
    * @param blockNum
    * @return
    */
    mapd_err_t deleteBlock(const int fileId, mapd_size_t *blockAddr);
    
   /**
    *
    * @param fileId
    * @param index
    * @return
    */
    mapd_err_t deleteBlock(const int fileId, const BlockAddrT &index);
    
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
    std::vector<FileInfoT> files_;     /**< The vector of files of chunks being managed. */
};

#endif // FILEMGR_H
