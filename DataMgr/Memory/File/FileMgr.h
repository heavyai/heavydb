/**
 * @file	FileMgr.h
 * @author	Steven Stewart <steve@map-d.com>
 *
 * This file includes the class specification for the FILE manager (FileMgr), and related
 * data structures and types.
 */
#ifndef DATAMGR_MEMORY_FILE_FILEMGR_H
#define DATAMGR_MEMORY_FILE_FILEMGR_H

#include <iostream>
#include <map>
#include <set>
#include "Block.h"
#include "FileBuffer.h"
#include "../AbstractDatum.h"
#include "../AbstractDataMgr.h"
#include "../../PgConnector/PgConnector.h"

using namespace Memory_Namespace;

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
        std::vector<Block*> blocks;			/// Block pointers for each block (including free blocks)
        std::set<mapd_size_t> freeBlocks; 	/// set of block numbers of free blocks
        
        /// Constructor
        FileInfo(const int fileId, FILE *f, const mapd_size_t blockSize, const mapd_size_t nblocks);
        
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
        
        /// Returns the number of free blocks available
        inline mapd_size_t numFreeBlocks() {
            return freeBlocks.size();
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
     * @brief A Chunk is the fundamental unit of execution in Map-D.
     *
     * A chunk is composed of logical blocks. These blocks can exist across multiple files managed by
     * the file manager.
     *
     * The collection of blocks is implemented as a FileBuffer object, which is composed of a vector of
     * MultiBlock objects, one for each logical block of the file buffer.
     */
    typedef FileBuffer Chunk;
    
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
     * @class   FileMgr
     * @brief
     */
    class FileMgr : public AbstractDataMgr { // implements
        
    public:
        /// Constructor
        FileMgr(std::string basePath = ".");
        
        /// Destructor
        virtual ~FileMgr();
        
        /// Creates a chunk with the specified key and page size.
        virtual void createChunk(const ChunkKey &key, mapd_size_t pageSize);
        
        /// Deletes the chunk with the specified key
        virtual void deleteChunk(const ChunkKey &key);
        
        /// Releases (frees) the memory used by the chunk with the specified key
        virtual void releaseChunk(const ChunkKey &key) {
            throw std::runtime_error("Operation not supported.");
        }
        
        /// Returns the a pointer to the chunk with the specified key.
        virtual AbstractDatum* getChunk(ChunkKey &key);
        
        /**
         * @brief Puts the contents of d into the Chunk with the given key.
         * @param key - Unique identifier for a Chunk.
         * @param d - An object representing the source data for the Chunk.
         * @return AbstractDatum*
         */
        virtual AbstractDatum* putChunk(const ChunkKey &key, AbstractDatum *d);
        
        // Datum API
        virtual AbstractDatum* createDatum(mapd_size_t pageSize, mapd_size_t nbytes);
        virtual void deleteDatum(AbstractDatum *d);
        virtual AbstractDatum* putDatum(AbstractDatum *d);
        
        /**
         * @brief Obtains free blocks -- creates new files if necessary -- of the requested size.
         *
         * Given a block size and number of blocks, this method updates the vector "blocks"
         * to include free blocks of the requested size. These blocks are immediately removed
         * from the free list of the affected file(s). If there are not enough blocks available
         * among current files, new files are created and their blocks are included in the vector.
         *
         * @param nblocks       The number of free blocks requested
         * @param blockSize     The size of each requested block
         * @param blocks        A vector containing the free blocks obtained by this method
         */
        void requestFreeBlocks(mapd_size_t nblocks, mapd_size_t blockSize, std::vector<Block> &blocks);
        
        /// Returns the current value of epoch
        inline int epoch() { return epoch_; }
        
    private:
        std::string basePath_; 				/// The OS file system path containing the files.
        std::vector<FileInfo*> files_;		/// A vector of files accessible via a file identifier.
        BlockSizeFileMMap fileIndex_; 		/// Maps block sizes to FileInfo objects.
        unsigned nextFileId_;				/// the index of the next file id
        int epoch_;                         /// the current epoch (time of last checkpoint)
        bool isDirty_;                      /// true if metadata changed since last writeState()
        
        ChunkKeyToChunkMap chunkIndex_; 	/// Index for looking up chunks
        std::map<ChunkKey, mapd_size_t> chunkBlockSize_; /// maps a Chunk to its block size
        
        PgConnector pgConnector_; /// Postgres connector for reading/writing file manager metadata
        
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
        
    };
    
} // File_Namespace

#endif // DATAMGR_MEMORY_FILE_FILEMGR_H
