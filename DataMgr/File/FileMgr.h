/**
 * @file    FileMgr.h
 * @author  Todd Mostak <todd@map-d.com>
 * @author  Steven Stewart <steve@map-d.com>
 * @brief   This file contains the class specification and related data structures for FileMgr.
 *
 * This file contains the FileMgr class specification, and the specifications of several related
 * types and data structures for use by the Map-D file management system. Each of these are documented
 * in this header file.
 *
 * The file manager manages files, which are collections of logical blocks. The types defined here are
 * designed to support the file manager's activities.
 *
 * Each file managed by file manager has a block size associated with it. Each block within the file
 * has the respective block size. A FileInfo object describes the metadata for a specific file.
 *
 * At present, metadata for the file manager is loaded from and stored into a Postgres database.
 * In the future, such metadata will be stored using the RDBMS's own infrastructure.
 *
 * @see File.h
 * @see Block.h
 */
#ifndef DATAMGR_FILE_FILEMGR_H
#define DATAMGR_FILE_FILEMGR_H

#include <iostream>
#include <memory>
#include <vector>
#include <map>
#include <set>
#include <utility>
#include <string>
#include "File.h"
#include "Block.h"
#include "../../Shared/types.h"
#include "../../Shared/errors.h"
#include "../PgConnector/PgConnector.h"

namespace File_Namespace {
    
    /**
     * @type FileInfo
     * @brief A FileInfo type has a file pointer and metadata about a file.
     *
     * A file info structure wraps around a file pointer in order to contain additional
     * information/metadata about the file that is pertinent to the file manager.
     *
     * The free blocks (freeBlocks) within a file must be tracked, and this is implemented using a
     * basic STL set. The set ensures that no duplicate blocks are included, and that the blocks
     * are sorted, faciliating the obtaining of consecutive free blocks by a constant time
     * pop operation, which may reduce the cost of DBMS disk accesses.
     *
     * Helper functions are provided: size(), available(), and used().
     */
    struct FileInfo {
        int fileId;							/// unique file identifier (i.e., used for a file name)
        FILE *f;							/// file stream object for the represented file
        mapd_size_t blockSize;				/// the fixed size of each block in the file
        mapd_size_t nblocks;				/// the number of blocks in the file
        std::vector<Block*> blocks;			/// vector of Block object pointers for each file block
        std::set<mapd_size_t> freeBlocks; 	/// set of block numbers of free blocks
        
        /// Constructor
        FileInfo(int fileId, FILE *f, mapd_size_t blockSize, mapd_size_t nblocks);
        
        /// Destructor
        ~FileInfo();
        
        /// Prints a summary of the file to stdout
        void print(bool blockSummary);
        
        /// Returns the number of bytes used by the file
        inline mapd_size_t size() {
            return blockSize * nblocks;
        }
        
        /// Returns the number of free bytes available
        inline mapd_size_t available() {
            return freeBlocks.size() * blockSize;
        }
        
        /// Returns the amount of used bytes; size() - available()
        inline mapd_size_t used() {
            return size() - available();
        }
    };
    
    /**
     * @type BlockSizeFileMMap
     * @brief Maps logical block sizes to files.
     *
     * The file manager uses this type in order to quickly find files of a certain block size.
     * A multimap is used to associate the key (block size) with values (file identifiers of files
     * having the matching block size).
     */
    typedef std::multimap<mapd_size_t, int> BlockSizeFileMMap;
    
    /**
     * @type Chunk
     * @brief A Chunk is the fundamental unit of execution in Map-D. It is a vector of MultiBlock pointers.
     *
     * A chunk is composed of logical blocks. These blocks can exist across multiple files managed by
     * the file manager.
     *
     * The collection of blocks is implemented as a set of MultiBlock* pointers. Since files contain
     * actual MultiBlock objects, it is better for a chunk to contain pointers to those objects in order
     * to avoid copy semantics and potential discrepancies between file blocks and chunk blocks.
     */
    typedef std::vector<MultiBlock*> Chunk;
    
    /**
     * @type ChunkKeyToChunkMap
     * @brief Maps ChunkKeys (unique ids for Chunks) to Chunk objects.
     *
     * The file system can store multiple chunks across multiple files. With that
     * in mind, the challenge is to be able to reconstruct the blocks that compose
     * a chunk upon request. A chunk key (ChunkKey) uniquely identifies a chunk,
     * and so ChunkKeyToChunkMap maps chunk keys to Chunk types, which are
     * vectors of MultiBlock* pointers (logical blocks).
     */
    typedef std::map<ChunkKey, Chunk> ChunkKeyToChunkMap;
    
    /**
     * @class FileMgr
     * @brief The file manager manages interactions between the DBMS and the file system.
     *
     * The main job of the file manager is to translate logical block addresses to physical
     * block addresses.
     *
     * FileMgr provides a chunk-level API, a block-level API for finer granularity, and a file-level
     * API. Care must be taken when using the different APIs such as not to "step on each others' toes"
     *
     * @todo asynch vs. synch?
     *
     */
    class FileMgr {
        
    public:
        /// Constructor with Postgress connector
        FileMgr(const std::string &basePath);
        
        /// Destructor
        ~FileMgr();
        
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
         * @return FileInfo* A pointer to the FileInfo object of the added file.
         */
        FileInfo* createFile(const mapd_size_t blockSize, const mapd_size_t nblocks);
        
        /**
         * @brief Finds the file in the file manager's FileMap (files_).
         *
         * @param fileId The unique file identifier of the file to be found.
         * @return A pointer to the found FileInfo object for the file.
         */
        FileInfo* getFile(const int fileId);
        
        /**
         * @brief Deletes a file from the file manager's repository.
         *
         * At a minimum, this method removes the file from both the FileMap (files_) and the
         * BlockSizeFileMap (blockSizeFile_). From the file manager's point of view, this removes
         * the file. Unless the value of destroy is "true," the actual physical file on disk
         * is not removed.
         *
         * Note that there may be residual references to the file in other data structures; thus,
         * it is advisable to check for the existence of the file prior to trying to access it.
         * This can be done by calling getFile().
         *
         * @param fileId The unique file identifier of the file to be deleted.
         * @param destroy If true, then the file on disk is deleted
         * @return An error code, should an error occur.
         */
        mapd_err_t deleteFile(const int fileId, const bool destroy = false);
        
        /**
         * @brief Returns the number of files currently being managed.
         * @return int The number of files currently being managed.
         */
        inline mapd_size_t fileCount() { return files_.size(); }
        
        /**
         * @brief Writes the specified number of bytes to the offset position into the file pointed at
         * by fInfo from src.
         *
         * @param fInfo address of the FileInfo object.
         * @param offset The location within the file where data is being written.
         * @param n The number of bytes to write to the file.
         * @param src The source buffer containing the data to be written.
         * @return err If not NULL, will hold an error code should an error occur.
         */
        mapd_err_t writeFile(FileInfo &fInfo, mapd_size_t offset, mapd_size_t n, mapd_addr_t src);
        
        /// Read n bytes of the file beginning at offset into buf
        mapd_err_t readFile(FileInfo &fInfo, mapd_size_t offset, mapd_size_t n, mapd_addr_t buf);
        
        /**
         * @brief Returns a pointer to a Block object for the specified block number in the file.
         *
         * @param fileId The unique file identifier of the file to be found.
         * @param blockNum The block number of the block to be retrieved.
         * @return A pointer to the found Block object.
         */
        Block* getBlock(const int fileId, mapd_size_t blockNum);
        
        /**
         * @brief Returns a pointer to a Block object for the specified block number in the file.
         *
         * @param FileInfo& A reference to the file that contains the block.
         * @param blockNum The block number of the block to be retrieved.
         * @return A pointer to the found BlockInfo object.
         */
        Block* getBlock(FileInfo &fInfo, mapd_size_t blockNum);
        
        /**
         * @brief Writes the contents of buf to the block.
         *
         * @param fileId
         * @param blockNum
         * @param n
         * @param buf
         */
        mapd_err_t putBlock(int fileId, mapd_size_t blockNum, mapd_addr_t buf);
        
        /**
         * @brief Writes the contents of buf to the block.
         *
         * @param fInfo
         * @param blockNum
         * @param n
         * @param buf
         */
        mapd_err_t putBlock(FileInfo &fInfo, mapd_size_t blockNum, mapd_addr_t buf);
        
        /**
         * @brief Clears the contents of a block in a file.
         *
         * This method clears the contents of a block in a file, resulting in the
         * endByteOffset being set to 0. The implementor may or may not modify the
         * actual block contents (@see FileMgr.cpp).
         *
         * If multiple versions of the block exist, then only the most recent will
         * be cleared, as it is assumed that the client wishes to update only
         * the most recent block.
         *
         * @param fileId The unique file identifier of the file containing the block.
         * @param blockNum The block number of the block to be cleared.
         * @return mapd_err_t An error code, should an error occur.
         */
        mapd_err_t clearBlock(const int fileId, mapd_size_t blockNum);
        
        /**
         * @brief Clears the contents of a block in a file.
         *
         * This method clears the contents of a block in a file, resulting in the
         * endByteOffset being set to 0. The implementor may or may not modify the
         * actual block contents (@see FileMgr.cpp).
         *
         * @param FileInfo& A reference to the the file containing the block.
         * @param blockNum The block number of the block to be cleared.
         * @return mapd_err_t An error code, should an error occur.
         */
        mapd_err_t clearBlock(FileInfo &fInfo, mapd_size_t blockNum);
        
        /**
         * @brief Adds the block to the list of free blocks.
         *
         * This method adds the block to the list of free blocks for a file.
         * It will call clearBlock() on the block prior to inserting it into
         * the free list.
         *
         * @param fileId The unique file identifier of the file to be found.
         * @param blockNum The block number of the block to be freed.
         * @return mapd_err_t An error code, should an error occur.
         */
        mapd_err_t freeBlock(const int fileId, mapd_size_t blockNum);
        
        /**
         * @brief Adds the block to the list of free blocks.
         *
         * This method adds the block to the list of free blocks for a file.
         * It will call clearBlock() on the block prior to inserting it into
         * the free list.
         *
         * @param fileId The unique file identifier of the file to be found.
         * @param blockNum The block number of the block to be freed.
         * @return mapd_err_t An error code, should an error occur.
         */
        mapd_err_t freeBlock(FileInfo &fInfo, mapd_size_t blockNum);
        
        /**
         *
         *
         * @param fileId
         * @param key
         * @param index
         * @return
         */
        Chunk* getChunkRef(const ChunkKey &key);
        
        /**
         * @param fileId
         * @param key
         * @param buf
         * @return
         */
        Chunk* getChunk(const ChunkKey &key, mapd_addr_t buf);
        
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
        mapd_err_t getChunkSize(const ChunkKey &key, mapd_size_t *nblocks, mapd_size_t *size);
        
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
        mapd_err_t getChunkActualSize(const ChunkKey &key, mapd_size_t *size);
        
        /**
         * Given a chunk key, this method writes to an existing chunk all of the data pointed to
         * by buf.
         */
        // @todo Comments to explain this in more depth are clearly "needed"
        mapd_err_t putChunk(const ChunkKey &key, mapd_size_t n, mapd_addr_t src, int epoch, mapd_size_t optBlockSize = -1);
        
        /**
         * Given a key, this method requests the file manager to create a new chunk of the
         * requested number of bytes (size). A pointer to the new Chunk object is returned,
         * or NULL upon failure.
         *
         * If src is NULL, then each block of the chunk will be empty; otherwise, the src
         * data will be copied into it.
         *
         * If the chunk already exists (based on looking up the key), or if the Chunk cannot
         * be created, then NULL is returned.
         * @todo proper-error handling for Chunk creation in FileMgr
         *
         * Note that epoch is only used if src is not NULL (i.e., src contains data to be
         * written to blocks, where epoch is stored in the block metadata).
         *
         * @param key   The unique identifier of the new chunk.
         * @param size  The amount of memory requested for the new chunk.
         * @param block Size The size of the logical disk blocks for which the chunk will be stored
         * @param src   The source data to be copied into the chunk (can be NULL).
         * @param epoch The epoch (time) at which the Chunk is being created.
         * @return A pointer to a new Chunk, or NULL.
         */
        Chunk* createChunk(const ChunkKey &key, const mapd_size_t n, const mapd_size_t blockSize, mapd_addr_t src, int epoch);
        
        /**
         * Given a chunk, this method deletes a chunk from the file system by freeing all
         * of the blocks associated with it.
         *
         * @param Chunk A reference to the chunk to be deleted.
         * @return MAPD_FAILURE or MAPD_SUCCESS
         */
        mapd_err_t deleteChunk(const ChunkKey &key);
        
        /**
         * @brief Returns the number of chunks in the file manager's chunk index.
         * @return int The number of chunks in the file manager's chunk index.
         */
        inline mapd_size_t chunkCount() { return chunkIndex_.size(); }
        
        /**
         * @brief Returns the base path used by the file manager.
         * @return std::string The base path used by the file manager.
         */
        inline std::string basePath() const { return basePath_; }
        
        /**
         * @brief Reads metadata into data structures from the Postgres database
         */
        void readState();
        
        /**
         * @brief Writes metadata from data structures to the Postgres database
         */
        void writeState();
        
        /**
         * @brief Clears (deletes) all metadata records from the Postgres database.
         */
        void clearState();
        
        /**
         * @brief Prints a representation of FileMgr's state to stdout
         */
        void print();

    private:
        std::string basePath_; 				/// The OS file system path containing the files. */
        std::vector<FileInfo*> files_;		/// A vector of files accessible via a file identifier. */
        BlockSizeFileMMap fileIndex_; 		/// Maps block sizes to FileInfo objects. */
        ChunkKeyToChunkMap chunkIndex_; 	/// Index for looking up chunks, which are vectors of MultiBlock */
        unsigned nextFileId_;				/// the index of the next file id */
        bool isDirty_;                      /// true if metadata changed since last writeState()
        
        /// Postgres connector object -- currently used for reading/writing file manager metadata
        PgConnector pgConnector_;
        
        /// Opens the FileInfo objects file handle. Returns MAPD_SUCCESS on success.
        inline mapd_err_t openFile(FileInfo& fInfo) {
            mapd_err_t err = MAPD_SUCCESS;
            if (!fInfo.f)
                fInfo.f = open(fInfo.fileId, &err);
            return err;
        }
        
        /**
         * @brief Inserts every block within the multiblock back into the free list, and then deletes the block
         */
        void freeMultiBlock(MultiBlock* mb);

    };
    
} // File_Namespace

#endif // DATAMGR_FILE_FILEMGR_H
