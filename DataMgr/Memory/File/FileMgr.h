/**
 * @file	FileMgr.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Todd Mostak <todd@map-d.com>
 *
 * This file includes the class specification for the FILE manager (FileMgr), and related
 * data structures and types.
 */

#ifndef DATAMGR_MEMORY_FILE_FILEMGR_H
#define DATAMGR_MEMORY_FILE_FILEMGR_H

#include <iostream>
#include <map>
#include <set>

#include "../../../Shared/global.h"
#include "Page.h"
#include "FileBuffer.h"
#include "FileInfo.h"
#include "../AbstractDatum.h"
#include "../AbstractDataMgr.h"
//#include "../../PgConnector/PgConnector.h"

using namespace Memory_Namespace;

namespace File_Namespace {

    /**
     * @type PageSizeFileMMap
     * @brief Maps logical page sizes to files.
     *
     * The file manager uses this type in order to quickly find files of a certain page size.
     * A multimap is used to associate the key (page size) with values (file identifiers of files
     * having the matching page size).
     */
    typedef std::multimap<mapd_size_t, int> PageSizeFileMMap;

    /**
     * @type Chunk
     * @brief A Chunk is the fundamental unit of execution in Map-D.
     *
     * A chunk is composed of logical pages. These pages can exist across multiple files managed by
     * the file manager.
     *
     * The collection of pages is implemented as a FileBuffer object, which is composed of a vector of
     * MultiPage objects, one for each logical page of the file buffer.
     */
    typedef FileBuffer Chunk;

    /**
     * @type ChunkKeyToChunkMap
     * @brief Maps ChunkKeys (unique ids for Chunks) to Chunk objects.
     *
     * The file system can store multiple chunks across multiple files. With that
     * in mind, the challenge is to be able to reconstruct the pages that compose
     * a chunk upon request. A chunk key (ChunkKey) uniquely identifies a chunk,
     * and so ChunkKeyToChunkMap maps chunk keys to Chunk types, which are
     * vectors of MultiPage* pointers (logical pages).
     */
    typedef std::map<ChunkKey, Chunk *> ChunkKeyToChunkMap;

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
        virtual AbstractDatum * createChunk(const ChunkKey &key, mapd_size_t pageSize);
        
        /// Deletes the chunk with the specified key
        virtual void deleteChunk(const ChunkKey &key);

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
        Page requestFreePage(mapd_size_t pagesize);

        void init();

        /**
         * @brief Obtains free pages -- creates new files if necessary -- of the requested size.
         *
         * Given a page size and number of pages, this method updates the vector "pages"
         * to include free pages of the requested size. These pages are immediately removed
         * from the free list of the affected file(s). If there are not enough pages available
         * among current files, new files are created and their pages are included in the vector.
         *
         * @param npages       The number of free pages requested
         * @param pagesize     The size of each requested page
         * @param pages        A vector containing the free pages obtained by this method
         */
        void requestFreePages(mapd_size_t npages, mapd_size_t pagesize, std::vector<Page> &pages);

        /**
         * @brief Fsyncs data files, writes out epoch and
         * fsyncs that
         */

        void checkpoint();
        /**
         * @brief Returns current value of epoch - should be
         * one greater than recorded at last checkpoint
         */
        inline int epoch() { return epoch_; }

        /**
         * @brief Returns FILE pointer associated with
         * requested fileId 
         *
         * @see FileBuffer
         */

        FILE * getFileForFileId(const int fileId);

    private:
        std::string basePath_; 				/// The OS file system path containing the files.
        std::vector<FileInfo*> files_;		/// A vector of files accessible via a file identifier.
        PageSizeFileMMap fileIndex_; 		/// Maps page sizes to FileInfo objects.
        unsigned nextFileId_;				/// the index of the next file id
        int epoch_;                         /// the current epoch (time of last checkpoint)
        FILE *epochFile_;
        bool isDirty_;                      /// true if metadata changed since last writeState()
        

        ChunkKeyToChunkMap chunkIndex_; 	/// Index for looking up chunks
        // #TM Not sure if we need this below


        /**
         * @brief Adds a file to the file manager repository.
         *
         * This method will create a FileInfo object for the file being added, and it will create
         * the corresponding file on physical disk with the indicated number of pages pre-allocated.
         *
         * A pointer to the FileInfo object is returned, which itself has a file pointer (FILE*) and
         * a file identifier (int fileId).
         *
         * @param fileName The name given to the file in physical storage.
         * @param pageSize The logical page size for the pages in the file.
         * @param numPages The number of logical pages to initially allocate for the file.
         * @return FileInfo* A pointer to the FileInfo object of the added file.
         */

        FileInfo* createFile(const mapd_size_t pageSize, const mapd_size_t numPages);
        FileInfo* openExistingFile(const std::string &path, const int fileId, const mapd_size_t pageSize, const mapd_size_t numPages, std::vector<HeaderInfo> &headerVec);
        void createEpochFile(const std::string &epochFileName);
        void openEpochFile(const std::string &epochFileName);
        void writeAndSyncEpochToDisk();
        
    };
    
} // File_Namespace

#endif // DATAMGR_MEMORY_FILE_FILEMGR_H
