#ifndef _FileMgr_h
#define _FileMgr_h

#include <vector>
#include <map>

#include "File.h"
#include "types.h"
#include "errors.h"

typedef std::vector <PageInfo> ChunkPages; // holds metadata for all pages in a chunk
typedef std::map <ChunkKey,ChunkPages> ChunkToPageMap; // ChunkKey is defined in types.h
typedef std::vector <bool> PageUsedVector;

struct PageAddress {
    int fileId;
    mapd_size_t filePageOffset; // in logical pages from beginning of file
}

struct PageInfo { // STEVE: Should struct member variables be appended with "_" as well?

    PageAddress pageAddress;
    mapd_size_t pageSize; // in bytes
    mapd_size_t endByteOffset; // in bytes from beginning of Page
    mapd_size_t headerSize; 
    int keySchemaId; // used to lookup header size
    //mapd_size_t headerSize; // don't necc need to keep this here - can have seperate
    unsigned int epoch;
    /* Next two variables to store location of old version of page if page is
     * updated so these can be freed at checkpoint time */
    bool wasDirtied; // should be false at checkpoint; any updates/inserts to page would make this true - meaning the metainfo for the page needs to be updated at checkpoint time.  This might be kept in a seperate data structure for fast (i.e. indexed) access at checkpoint time
    int oldFileId; //oldFileId = -1 by default to signifiy that page has not been changed 
    mapd_size_t oldFilePageOffset;
};

struct PageHeader { //used to read header into 
    int bytesWritten; //made int so -1 can signify free page
    unsigned int epoch; 
    int pageId; //Assumed to be zero based
    ChunkKey chunkKey;
}; 

struct FileInfo { 
    File file;
    PageUsedVector pageUsedVector; // stores a bit (or boolean) for each page specifying if it is free
};



struct

class FileMgr {

    public:
        mapd_err_t checkPoint(); // Runs a checkpoint. This involves:
                                 // 1. Telling buffer pool to flush all dirty
                                 // pages (maybe done externally so FileManager
                                 // does not need reference to buffer manager
                                 // 2. Marking as free old versions of all
                                 //    pages that have been updated since last
                                 //    epoch
                                 // 3. Incrementing epoch counter and flushing
                                 //    this to disk

        mapd_err_t getChunkSize (const ChunkKey &chunkKey, mapd_size_t &chunkSize) const; //Returns byte size of chunk (truncated to last record)
        mapd_err_t copyChunkToMemory(const ChunkKey &chunkKey, void *mem) const; //Always copies newest version of pages in chunk in contiguous pageId order to memory - usually requested by buffer pool
        mapd_err_t deleteChunk (const ChunkKey &chunkKey); // Needs to update chunkToPageMap - can't free page until    
        mapd_err_t flushPageToDisk (const ChunkKey &chunkKey, const int pageId, const void *mem); //can you flush a new page or do you have to create it first
        mapd_err_t createPage (const ChunkKey &chunkKey, int &pageId); // returns pageId of newly created page by reference




    private:

        std::string basePath_; 
        std::vector<FileInfo> files_;
        std::vector<mapd_size_t> keySchemaSizes_; // stores size of keySchema headers in bytes - implicitly ordered by KeySchema id - starting at 0
        std::vector<int> d
        ChunkToPageMap chunkToPageMap_;
        PageUsedVector pageUsedVector_;
        unsigned int epoch_; // where is this stored on disk - in file header?

        mapd_err_t readHeaderFromFile (const int fileId, const int filePageOffset, PageHeader &pageHeader) const;
        mapd_err_t getFilenameFromId(const int fileId, std::string &fileName) const; // returns filename by reference
        mapd_err_t getNumDatFilesInDir(const string &basePath, mapd_size_t &numDataFiles) const; //base path may but does not have to be class member variable basePath_






};

#endif // _FileMgr_h
