/**
 * @file    FileMgr.cpp
 * @author  Steven Stewart <steve@map-d.com>
 *
 * Implementation file for the file manager.
 *
 * @see FileMgr.h
 */
#include <iostream>
#include <cassert>
#include <cstdio>
#include <algorithm>
#include <string>
#include <sstream>
#include <vector>
#include <cassert>
#include <exception>
#include "FileMgr.h"
#include "../../Shared/global.h"

using std::vector;

namespace File_Namespace {
    
    FileInfo::FileInfo(int fileIdIn, FILE *fIn, mapd_size_t blockSizeIn, mapd_size_t nblocksIn)
    : fileId(fileIdIn), f(fIn), blockSize(blockSizeIn), nblocks(nblocksIn)
    {
        // initialize blocks and free block list
        for (mapd_size_t i = 0; i < nblocks; ++i) {
            blocks.push_back(new Block(fileId, i));
            freeBlocks.insert(i);
        }
    }
    
    FileInfo::~FileInfo() {
        // free memory used by Block objects
        for (mapd_size_t i = 0; i < blocks.size(); ++i)
            delete blocks[i];
        
        // close file, if applicable
        if (f && close(f) != MAPD_SUCCESS)
            fprintf(stderr, "[%s:%d] Error closing file %d.\n", __func__, __LINE__, fileId);
    }
    
    void FileInfo::print(bool blockSummary) {
        printf("File #%d", fileId);
        printf(" size = %lu", size());
        printf(" used = %lu", used());
        printf(" free = %lu", available());
        printf("\n");
        if (!blockSummary)
            return;
        
        for (mapd_size_t i = 0; i < blocks.size(); ++i) {
            // @todo block summary
        }
    }
    
    FileMgr::FileMgr(const std::string &basePath) : basePath_(basePath), pgConnector_("mapd", "mapd"), isDirty_(false), nextFileId_(0)
    {
        mapd_err_t status;
        
        // Create FileInfo table for storing metadata
        status = pgConnector_.query("CREATE TABLE IF NOT EXISTS FileInfo(file_id integer PRIMARY KEY, block_size integer, nblocks integer)");
        assert(status == MAPD_SUCCESS);
        
        // Create fileinfo_blocks table
        status = pgConnector_.query("CREATE TABLE IF NOT EXISTS FileInfo_Blocks(file_id integer not null, block_num integer not null, used integer not null, PRIMARY KEY(file_id, block_num));");
        assert(status == MAPD_SUCCESS);
        
        // Create multiblock table
        status = pgConnector_.query("CREATE TABLE IF NOT EXISTS MultiBlock(MultiBlock_id integer not null, version integer not null, epoch integer not null, file_id integer not null, block_num INT not null, PRIMARY KEY(MultiBlock_id, version));");
        assert(status == MAPD_SUCCESS);
        
        // Create chunk_multiblock table
        status = pgConnector_.query("CREATE TABLE IF NOT EXISTS Chunk_MultiBlock(ChunkKey integer[], MultiBlock_id integer, PRIMARY KEY(ChunkKey, MultiBlock_id));");
        assert(status == MAPD_SUCCESS);
        
        // read in metadata and update internal data structures
        readState();
    }
    
    FileMgr::~FileMgr() {
        // write file manager metadata to postgres database
        writeState();
        
        // free memory used by FileInfo objects
        for (int i = 0; i < files_.size(); ++i)
            delete files_[i];
        
        // free memory allocated for MultiBlock objects for each Chunk
        for(auto it = chunkIndex_.begin(); it != chunkIndex_.end(); ++it) {
            Chunk &v = (*it).second;
            for (auto it2 = v.begin(); it2 != v.end(); ++it2)
                delete *it2;
        }
    }
    
    FileInfo* FileMgr::createFile(const mapd_size_t blockSize, const mapd_size_t nblocks) {
        if (blockSize == 0 || nblocks == 0) {
            // @todo proper exception handling would be desirable, eh
            return NULL;
        }
        
        // create the new file
        FILE *f = NULL;
        f = create(nextFileId_, blockSize, nblocks, NULL);
        
        // check for error
        if (f == NULL) {
            fprintf(stderr, "[%s:%d] Error: unable to create file.\n", __func__, __LINE__);
            return NULL;
        }
        
        // update file manager data structures
        int fileId = nextFileId_;
        nextFileId_++;
        
        FileInfo *fInfo = NULL;
        try {
            fInfo = new FileInfo(fileId, f, blockSize, nblocks);
            files_.push_back(fInfo);
            fileIndex_.insert(std::pair<mapd_size_t, int>(blockSize, fileId));
        }
        catch (const std::bad_alloc& e) {
            std::cout << "Bad allocation exception encountered: " << e.what() << std::endl;
            return NULL;
        }
        catch (const std::exception& e) {
            std::cout << "Exception encountered: " << e.what() << std::endl;
            if (!fInfo) delete fInfo;
            return NULL;
        }
        assert(files_.back() == fInfo);
        return fInfo;
    }
    
    FileInfo* FileMgr::getFile(const int fileId) {
        if (fileId < 0 || fileId > files_.size())
            return NULL;
        return files_[fileId];
    }
    
    mapd_err_t FileMgr::deleteFile(const int fileId, const bool destroy) {
        
        // confirm the file exists and obtain pointer
        FileInfo *fInfo = getFile(fileId);
        if (!fInfo)
            return MAPD_FAILURE;
        
        // remove the file from the fileIndex_
        BlockSizeFileMMap::iterator it = fileIndex_.lower_bound(fInfo->blockSize);
        for (it = fileIndex_.begin(); it != fileIndex_.end(); ++it) {
            if (it->second == fileId)
                break;
        }
        if (it != fileIndex_.end())
            fileIndex_.erase(it);
        
        // remove the file from the vector of files_
        files_.erase(files_.begin() + fileId);
        
        // @todo error-checking if erase fails?
        // @todo physically delete the file on disk
        return MAPD_SUCCESS;
    }
    
    // Gil wrote this. Send any complaints to Map-D's Zurich office.
    mapd_err_t FileMgr::readFile(FileInfo &fInfo, mapd_size_t offset, mapd_size_t n, mapd_addr_t buf) {
        mapd_err_t err = MAPD_SUCCESS;
        size_t result = read(fInfo.f, offset, n, buf, &err);
        if (result != n)
            err = MAPD_FAILURE;
        // @todo proper error handling
        return err;
    }
    
    mapd_err_t FileMgr::writeFile(FileInfo &fInfo, mapd_size_t offset, mapd_size_t n, mapd_addr_t src) {
        //size_t write(FILE *f, mapd_addr_t offset, mapd_size_t n, mapd_addr_t buf, mapd_err_t *err);
        mapd_err_t err = MAPD_SUCCESS;
        size_t result = write(fInfo.f, offset, n, src, &err);
        return err;
    }
    
    
    Block* FileMgr::getBlock(const int fileId, mapd_size_t blockNum) {
        FileInfo *fInfo = FileMgr::getFile(fileId);
        return !fInfo ? NULL : getBlock(*fInfo, blockNum);
    }
    
    Block* FileMgr::getBlock(FileInfo &fInfo, mapd_size_t blockNum) {
        assert(blockNum < fInfo.blocks.size() && fInfo.blocks[blockNum]);
        return fInfo.blocks[blockNum];
    }
    
    mapd_err_t FileMgr::putBlock(int fileId, mapd_size_t blockNum, mapd_addr_t buf) {
        FileInfo *fInfo;
        return ((fInfo = getFile(fileId)) == NULL) ? MAPD_FAILURE : putBlock(*fInfo, blockNum, buf);
    }
    
    mapd_err_t FileMgr::putBlock(FileInfo &fInfo, mapd_size_t blockNum, mapd_addr_t buf) {
        // assert buf
        
        // open the file if it is not open already
        if (openFile(fInfo) != MAPD_SUCCESS) {
            printf("openfile error");
            return MAPD_FAILURE;
        }
        // write the block to the file
        mapd_err_t err;
        size_t wrote = writeBlock(fInfo.f, fInfo.blockSize, blockNum, buf, &err);
        assert(wrote == fInfo.blockSize);
        
        return err;
    }
    
    mapd_err_t FileMgr::clearBlock(const int fileId, mapd_size_t blockNum) {
        FileInfo *fInfo = FileMgr::getFile(fileId);
        return !fInfo ? MAPD_FAILURE : clearBlock(*fInfo, blockNum);
    }
    
    mapd_err_t FileMgr::clearBlock(FileInfo &fInfo, mapd_size_t blockNum) {
        Block *b = getBlock(fInfo, blockNum);
        if (b) {
            b->used = 0;
            return MAPD_SUCCESS;
        }
        return MAPD_FAILURE;
    }
    
    mapd_err_t FileMgr::freeBlock(const int fileId, mapd_size_t blockNum) {
        FileInfo *fInfo = getFile(fileId);
        return !fInfo ? MAPD_FAILURE : freeBlock(*fInfo, blockNum);
    }
    
    mapd_err_t FileMgr::freeBlock(FileInfo &fInfo, mapd_size_t blockNum) {
        mapd_err_t err = MAPD_SUCCESS;
        err = clearBlock(fInfo, blockNum);
        if (err == MAPD_SUCCESS)
            fInfo.freeBlocks.insert(blockNum); // @todo error-checking on insert() ?
        return err;
    }
    
    Chunk* FileMgr::getChunkRef(const ChunkKey &key) {
        auto it = chunkIndex_.find(key);
        return it != chunkIndex_.end() ? &it->second : NULL;
    }
    
    Chunk* FileMgr::getChunk(const ChunkKey &key, mapd_addr_t buf) {
        assert(buf);
        
        // find chunk
        auto it = chunkIndex_.find(key);
        if (it == chunkIndex_.end()) // chunk doesn't exist
            return NULL;
        
        // copy contents of chunk to buf
        Chunk &c = it->second;
        for (int i = 0; i < c.size(); ++i) {
            
            // get most recent address of current block
            Block &blk = c[i]->current();
            
            // obtain a reference to the file of the block address
            int fileId = blk.fileId;
            FileInfo *fInfo = getFile(fileId);
            if (!fInfo)
                return NULL;
            
            // open the file if it is not open already
            if (openFile(*fInfo) != MAPD_SUCCESS)
                return NULL;
            
            // read block from file into buf
            mapd_err_t err;
            mapd_size_t used = c[i]->current().used;
            read(fInfo->f, blk.blockNum * c[i]->blockSize, used, buf, &err);
            buf += used;
            // @todo should ensure no gaps in blocks of Chunk
            
            if (err != MAPD_SUCCESS)
                return NULL;
        }
        return &c;
    }
    
    mapd_err_t FileMgr::getChunkSize(const ChunkKey &key, mapd_size_t *nblocks, mapd_size_t *size) {
        assert(size || nblocks); // at least one of these should be not NULL
        mapd_err_t err = MAPD_SUCCESS;
        
        ChunkKeyToChunkMap::iterator iter = chunkIndex_.find(key);
        if (iter == chunkIndex_.end()) {
            // not found
            err = MAPD_ERR_CHUNK_NOT_FOUND; // chunk doesn't exist
            return err;
        }
        
        // found
        Chunk &c = iter->second;
        
        // check if chunk has no blocks
        if (nblocks) {
            *nblocks = c.size();
            if (*nblocks < 1) {
                if (size) *size = 0;
                return err;
            }
        }
        if (size) { // Compute size based on block sizes
            *size = 0;
            for (int i = 0; i < c.size(); ++i)
                *size += c[i]->blockSize;
        }
        return err;
    }
    
    mapd_err_t FileMgr::getChunkActualSize(const ChunkKey &key, mapd_size_t *size) {
        assert(size);
        mapd_err_t err = MAPD_SUCCESS;
        
        ChunkKeyToChunkMap::iterator iter = chunkIndex_.find(key);
        if (iter == chunkIndex_.end()) // not found
            return MAPD_ERR_CHUNK_NOT_FOUND;
        
        // Compute size based on actual bytes used in block
        Chunk &c = iter->second;
        for (int i = 0; i < c.size(); ++i)
            *size += c[i]->current().used;
        
        return err;
    }
    
    mapd_err_t FileMgr::putChunk(const ChunkKey &key, mapd_size_t size, mapd_addr_t src, int epoch, mapd_size_t optBlockSize) {
        assert(src);
        mapd_err_t err = MAPD_SUCCESS;
        
        // ensure chunk exists
        Chunk* c;
        if ((c = getChunkRef(key)) == NULL) { // not found
            fprintf(stderr, "getChunkRef failed\n");
            return MAPD_ERR_CHUNK_NOT_FOUND;
        }
        
        mapd_size_t blockSize;
        
        // obtain blockSize from Chunk. if no blocks in the Chunk, use default param.
        if (c->size() == 0) {
            if (optBlockSize == -1)  {
                fprintf(stderr, "[%s:%d] Notice: using Map-D default block size.\n", __FILE__, __LINE__);
                blockSize = MAPD_DEFAULT_BLOCK_SIZE;
            }
            else
                blockSize = optBlockSize;
        }
        else {
            // Otherwise, obtain FileInfo object of first block in order
            // to get the block size
            Block &blk = (*c)[0]->current();
            blockSize = getFile(blk.fileId)->blockSize;
        }
        
        // number of blocks to be added from src
        mapd_size_t nblocks = (size + blockSize - 1) / blockSize;
        
        // blockCount: number of blocks written so far
        mapd_size_t blockCount = 0;
        
        // Obtain an iterator over files having the desired block size to be written
        auto it = fileIndex_.lower_bound(blockSize);
        
        // Write blockSize bytes to a new version of each existing logical block of the Chunk
        for (int i = 0; i < c->size(); ++i) {
            
            // find a suitable file (i.e., having the desired block size)
            FileInfo* fInfo = NULL;
            for (; it != fileIndex_.end(); ++it)
                if (getFile(it->second)->available() > 0)
                    fInfo = getFile(it->second);
            it--; // preserve iterator position
            
            if (fInfo == NULL) {
                // create a new file with the default number of blocks
                fInfo = createFile(blockSize, MAPD_DEFAULT_N_BLOCKS);
            }
            assert(fInfo->freeBlocks.size() > 0);
            
            // obtain first available free block number, and remove it from free block list
            mapd_size_t freeBlockNum;
            auto itFree = fInfo->freeBlocks.begin();
            freeBlockNum = *itFree;
            fInfo->freeBlocks.erase(itFree);
            
            // Push the previously free block to be used as the new version
            // of the current MultiBlock with the specified epoch
            (*c)[i]->push(getBlock(*fInfo, freeBlockNum), epoch);
            
            // Write the correct block of src to the identified free block in fInfo
            mapd_size_t bytesWritten = write(fInfo->f, freeBlockNum*fInfo->blockSize, blockSize, src+blockCount*blockSize, &err);
            
            nblocks--;
            blockCount++;
        }
        
        // Create new MultiBlock objects for the Chunk for remaining unwritten data
        while (nblocks > 0) {
            
            // find a suitable file (i.e., having the desired block size)
            FileInfo* fInfo = NULL;
            for (; it != fileIndex_.end(); ++it)
                if (getFile(it->second)->available() > 0)
                    fInfo = getFile(it->second);
            it--; // preserve iterator position
            
            if (fInfo == NULL) {
                // create a new file with the default number of blocks
                fInfo = createFile(blockSize, MAPD_DEFAULT_N_BLOCKS);
            }
            assert(fInfo->freeBlocks.size() > 0);
            
            // obtain first available free block number, and remove it from free block list
            mapd_size_t freeBlockNum;
            auto itFree = fInfo->freeBlocks.begin();
            freeBlockNum = *itFree;
            fInfo->freeBlocks.erase(itFree);
            
            MultiBlock* mb = new MultiBlock(fInfo->blockSize);
            mb->push(getBlock(*fInfo, freeBlockNum), epoch);
            c->push_back(mb);
            
            mapd_size_t bytesWritten = write(fInfo->f, freeBlockNum*fInfo->blockSize, blockSize, src+blockCount*blockSize, &err);
            
            nblocks--;
            blockCount++;
            
        } // while (nblocks > 0)
        
        return err;
    }
    
    // Inserts free blocks into the chunk; creates a new file if necessary
    Chunk* FileMgr::createChunk(const ChunkKey &key, const mapd_size_t size, const mapd_size_t blockSize, mapd_addr_t src, int epoch) {
        // check if the chunk already exists based on key
        Chunk *ctmp = NULL;
        if ((ctmp = getChunkRef(key)) != NULL) {
            fprintf(stderr, "Warning: Chunk for specified key already exists. Using existing Chunk.\n");
            return ctmp;
        }
        
        // Otherwise, add an entry to the file manager's chunk index
        chunkIndex_.insert(std::pair<ChunkKey, Chunk>(key, Chunk()));
        
        // Call putChunk to copy src into the new Chunk
        mapd_err_t err = MAPD_SUCCESS;
        if (src != NULL)
            err = putChunk(key, size, src, epoch, blockSize);
        
        // if putChunk() fails, then remove key from chunkIndex and return NULL
        if (err != MAPD_SUCCESS) {
            fprintf(stderr, "[%s:%s:%d] Error: unable to create chunk.\n", __FILE__, __func__, __LINE__);
            chunkIndex_.erase(key);
            return NULL;
        }
        
        // success!
        return getChunkRef(key);
    }
    
    void FileMgr::freeMultiBlock(MultiBlock* mb) {
        while (mb->version.size() > 0) {
            //get fileInfo of each block
            Block &blk = *mb->version.front();
            FileInfo *fInfo = getFile(blk.fileId);
            
            // expression refers to file offset of most recent block in MultiBlock
            fInfo->freeBlocks.insert(blk.blockNum);
            // delete the front block
            mb->pop();
        }
        //now, delete the whole multiblock
        delete mb;
    }
    
    mapd_err_t FileMgr::deleteChunk(const ChunkKey &key) {
        Chunk* c = NULL;
        
        // ensure the Chunk exists
        if ((c = getChunkRef(key)) == NULL)
            return MAPD_FAILURE;
        
        // While there are still multiblocks in the chunk, pop the back and free it
        while (c->size() > 0) {
            freeMultiBlock(c->back());
            c->pop_back();
        }
        
        // Remove Chunk from ChunkIndex. Return failure if it does not remove exactly one chunk.
        if (chunkIndex_.erase(key) != 1)
            return MAPD_FAILURE;
        
        return MAPD_SUCCESS;
    }
    
    void FileMgr::print() {
        printf("FileMgr: %p, %lu files\n", this, files_.size());
        
        // files_ summary (FileInfo objects)
        for (int i = 0; i < files_.size(); ++i)
            printf("File: id=%d blockSize=%lu nblocks=%lu\n", files_[i]->fileId, files_[i]->blockSize, files_[i]->nblocks);
        
        // fileIndex_ summary (maps block sizes to file ids)
        for (auto it = fileIndex_.begin(); it != fileIndex_.end(); ++it)
            printf("[%lu]->%d\n", (*it).first, (*it).second);
        
        // chunkIndex_ summary (ChunkKey mapped to MultiBlocks)
        for (auto it = chunkIndex_.begin(); it != chunkIndex_.end(); ++it) {
            ChunkKey key = (*it).first;
            printf("<");
            for (int i = 0; i < key.size(); ++i)
                printf("%d%s", key[i], i+1 < key.size() ? "," : "");
            printf("> %lu multiblocks\n", (*it).second.size());
        }
    }
    
    /**
     * This method populates FileMgr's metadata data structures, which include:
     *      files_      An array of pointers to FileInfo objects
     *      fileIndex_  A multimap mapping file block sizes to file ids
     *      chunkIndex_ A map from ChunkKey to Chunk (vector of MultiBlock)
     *
     *  Additionally, nextFileId_ should be set to exactly 1 higher than the highest file
     *  id of those read from the Postgres table.
     *
     *  This data structures are populated by querying a Postgres database via the pgConnector
     *  object. The .query() method executes a query, and .getData() retrieves the results.
     */
    void FileMgr::readState() {
        FILE *f = nullptr;          // the file handle passed to a new FileInfo object
        std::vector<int> file_ids;  // metadata: unique file identifier
        mapd_size_t block_size;     // metadata: block size for the file
        mapd_size_t nblocks;        // metadata: number of blocks in the file
        std::string q1;             // query text
        size_t numRows;             // number of rows in the result set of the query

        // READ METADATA: FileInfo metadata
        q1 = "select file_id, block_size, nblocks from FileInfo order by file_id";
        mapd_err_t status = pgConnector_.query(q1);
        assert(status == MAPD_SUCCESS);
        numRows = pgConnector_.getNumRows();
        
        for (int i = 0; i < numRows; ++i) {
            
            // retrieve metadata for new FileInfo object
            file_ids.push_back(pgConnector_.getData<int>(i,0));
            f = open(file_ids.back(), NULL);
            block_size = pgConnector_.getData<int>(i,1);
            nblocks = pgConnector_.getData<int>(i,2);
            
            // nextFileId_ should be one greater than the max file_id of those read
            this->nextFileId_ = file_ids.back() > this->nextFileId_ ? file_ids.back()+1 : nextFileId_+1;
            
            // insert the new FileInfo object into this->files_
            FileInfo *fInfo = new FileInfo(file_ids.back(), f, block_size, nblocks);
            files_.push_back(fInfo);

            // insert an entry in this->fileIndex_ for this file
            fileIndex_.insert(std::pair<mapd_size_t, int>(block_size, file_ids.back()));
        }
        
        // READ METADATA: FileInfo_Blocks
        // For each FileInfo object, read in its block metadata
        for (int i = 0; i < files_.size(); i++) {
            FileInfo *fInfo = files_[i];
            q1 = "select block_num, used from fileinfo_blocks where file_id = " + std::to_string(fInfo->fileId) + " order by block_num asc;";
            status = pgConnector_.query(q1);
            assert(status == MAPD_SUCCESS);
            numRows = pgConnector_.getNumRows();
            
            for (int j = 0; j < numRows; ++j) { // numRows is number of blocks for current file
                Block *b = fInfo->blocks[j];
                b->fileId = fInfo->fileId;
                b->blockNum = pgConnector_.getData<int>(j,0);
                b->used = pgConnector_.getData<int>(j,1);
            }
        }
        
        // READ METADATA: Chunk
        q1 = "select array_to_string(chunkkey, ',') from chunk_multiblock group by chunkkey";
        status = pgConnector_.query(q1);
        assert(status == MAPD_SUCCESS);
        numRows = pgConnector_.getNumRows();

        // will contain ChunkKey entries
        std::vector<std::vector<int>> chunkkeys;
        
        // populate chunkkeys vector with keys obtained from the result set
        for (int i = 0; i < numRows; ++i) {
            
            // key is initially read in as a string
            std::string strKey = pgConnector_.getData<std::string>(i, 0);
            
            // strKey is parsed, creating the ChunkKey (vector of int) called key
            std::vector<int> key;
            std::stringstream ss(strKey);
            std::string item;
            while (std::getline(ss, item, ','))
                key.push_back(atoi(item.c_str()));
            
            // finally, the key is pushed into the vector of chunkkeys
            chunkkeys.push_back(key);
        }
        
        // for each chunkkey, insert Chunk into chunkIndex_ and query for its multiblocks
        for (int i = 0; i < chunkkeys.size(); ++i) {

            // insert Chunk into chunkIndex_
            chunkIndex_.insert(std::pair<ChunkKey, Chunk>(chunkkeys[i], Chunk()));
            Chunk *c = &chunkIndex_[chunkkeys[i]];
            assert(c);
            
            // build query to obtain multiblock metadata for current Chunk
            q1 = "select multiblock_id, version, epoch, file_id, block_num from multiblock where multiblock_id in (select multiblock_id from chunk_multiblock where chunkkey = '{";
            for (int j = 0; j < chunkkeys[i].size(); ++j) {
                q1 += std::to_string(chunkkeys[i][j]);
                if (j+1 < chunkkeys[i].size()) q1 += ",";
            }
            q1 += "}') order by version asc;";
            printf("q1 = %s\n", q1.c_str());
            
            // execute query
            status = pgConnector_.query(q1);
            assert(status == MAPD_SUCCESS);
            numRows = pgConnector_.getNumRows();
            
            // for each multiblock, create an entry in chunkIndex_
            for (int i = 0; i < numRows; ++i) {
                // get multiblock metadata
                // int version = pgConnector_.getData<int>(i,0);
                int epoch = pgConnector_.getData<int>(i,1);
                int file_id = pgConnector_.getData<int>(i,2);
                int block_num = pgConnector_.getData<int>(i,3);
                
                print();
                
                // create a multiblock for current chunk using
                assert(file_id < files_.size());
                c->push_back(new MultiBlock(files_[file_id]->blockSize));
                
                // insert a new block into the multiblock
                c->back()->push(new Block(file_id, block_num), epoch);
            }
            print();
        }
    }
    
    void FileMgr::writeState() {
        std::string query1, query2, query3;
        mapd_err_t status;

        // CLEAR THE EXISTING METADATA TABLES
        return;
        clearState();
        
        // WRITE METADATA: FileInfo
        for (int i = 0; i < files_.size(); ++i) {
            FileInfo *fInfo = files_[i];
            //printf("i=%d fInfo->fileId=%d\n", i, fInfo->fileId);
            query2 = "insert into fileinfo(file_id, block_size, nblocks) values (" + std::to_string(fInfo->fileId) + ", " + std::to_string(fInfo->blockSize) + ", " + std::to_string(fInfo->nblocks) + ");";
            //printf("query2 = %s\n", query2.c_str());
            status = pgConnector_.query(query2);
            assert(status == MAPD_SUCCESS);
            
            // WRITE METADATA: FileInfo_blocks
            for (int j = 0; j < fInfo->nblocks; ++j) {
                Block *b = fInfo->blocks[j];
                query3 = "insert into fileinfo_blocks(file_id, block_num, used) values (" + std::to_string(b->fileId) + ", " + std::to_string(b->blockNum) + ", " + std::to_string(b->used) + ");";
                status = pgConnector_.query(query3);
                assert(status == MAPD_SUCCESS);
            }
        }
    }
    
    void FileMgr::clearState() {
        std::string q1;
        mapd_err_t status;

        // CLEAR THE EXISTING METADATA TABLES
        q1 = "delete from fileinfo;";
        status = pgConnector_.query(q1);
        assert(status == MAPD_SUCCESS);
        
        q1 = "delete from fileinfo_blocks;";
        status = pgConnector_.query(q1);
        assert(status == MAPD_SUCCESS);
        
        q1 = "delete from multiblock;";
        status = pgConnector_.query(q1);
        assert(status == MAPD_SUCCESS);
        
        q1 = "delete from chunk_multiblock;";
        status = pgConnector_.query(q1);
        assert(status == MAPD_SUCCESS);
    }
    
} // File_Namespace
