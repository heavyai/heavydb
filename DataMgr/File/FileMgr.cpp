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
    
    FileInfo::FileInfo(const int fileId, FILE *f, const mapd_size_t blockSize, mapd_size_t nblocks)
    : fileId(fileId), f(f), blockSize(blockSize), nblocks(nblocks)
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
        if (f)
            close(f);
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
    
    FileMgr::FileMgr(const std::string &basePath) : basePath_(basePath), pgConnector_("mapd", "mapd"), isDirty_(false), nextFileId_(0), epoch_(0)
    {
        mapd_err_t status;
        
        // Create FileInfo table for storing metadata
        status = pgConnector_.query("CREATE TABLE IF NOT EXISTS FileInfo(file_id integer PRIMARY KEY, block_size integer, nblocks integer)");
        assert(status == MAPD_SUCCESS);
        
        // Create fileinfo_blocks table
        status = pgConnector_.query("CREATE TABLE IF NOT EXISTS FileInfo_Blocks(file_id integer not null, block_num integer not null, used integer not null, PRIMARY KEY(file_id, block_num));");
        assert(status == MAPD_SUCCESS);
        
        // Create chunk_multiblock table
        status = pgConnector_.query("CREATE TABLE IF NOT EXISTS Chunk_MultiBlock(ChunkKey integer[], MultiBlock_id integer, PRIMARY KEY(ChunkKey, MultiBlock_id));");
        assert(status == MAPD_SUCCESS);
        
        // Create multiblock table
        status = pgConnector_.query("CREATE TABLE IF NOT EXISTS MultiBlock(multiblock_id integer not null, block_id integer not null, epoch integer not null, file_id integer not null, block_num INT not null, PRIMARY KEY(MultiBlock_id, block_id));");
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

        // check arguments
        if (blockSize == 0 || nblocks == 0)
            throw std::invalid_argument("blockSize and nblocks must be greater than 0.");
        
        // create the new file
        FILE *f = create(nextFileId_, blockSize, nblocks);
        if (f == nullptr)
            throw std::runtime_error("Unable to create the new file.");
        
        // instantiate a new FileInfo for the newly created file
        int fileId = nextFileId_++;
        FileInfo *fInfo = new FileInfo(fileId, f, blockSize, nblocks);
        assert(fInfo);

        // update file manager data structures
        files_.push_back(fInfo);
        fileIndex_.insert(std::pair<mapd_size_t, int>(blockSize, fileId));
        
        assert(files_.back() == fInfo); // postcondition
        return fInfo;
    }
    
    FileInfo* FileMgr::getFile(const int fileId) {
        if (fileId < 0 || fileId >= files_.size())
            throw std::invalid_argument("fileId not within required range");
        return files_[fileId];
    }
    
    void FileMgr::deleteFile(const int fileId, const bool destroy) {
        // confirm the file exists and obtain pointer
        FileInfo *fInfo = getFile(fileId);
        if (!fInfo)
            throw std::runtime_error("Requested file does not exist.");
        
        // remove the file from the fileIndex_
        BlockSizeFileMMap::iterator fileIt = fileIndex_.lower_bound(fInfo->blockSize);
        for (fileIt = fileIndex_.begin(); fileIt != fileIndex_.end(); ++fileIt) {
            if (fileIt->second == fileId)
                break;
        }
        if (fileIt != fileIndex_.end())
            fileIndex_.erase(fileIt);
        
        // remove the file from the vector of files_
        files_.erase(files_.begin() + fileId);
        
        // @todo physically delete the file on disk
        // @todo what is the impact on Chunk metadata when deleting a file?
    }
    
    size_t FileMgr::writeFile(FileInfo &fInfo, const mapd_size_t offset, const mapd_size_t size, mapd_addr_t buf) {
        size_t result = write(fInfo.f, offset, size, buf);
        return result;
    }

    size_t FileMgr::readFile(FileInfo &fInfo, const mapd_size_t offset, const mapd_size_t size, mapd_addr_t buf) {
        size_t bytesRead = read(fInfo.f, offset, size, buf);
        return bytesRead;
    }
    
    Block* FileMgr::getBlock(const int fileId, const mapd_size_t blockNum) {
        return getBlock(FileMgr::getFile(fileId), blockNum);
    }
    
    Block* FileMgr::getBlock(FileInfo *fInfo, const mapd_size_t blockNum) {
        assert(blockNum < fInfo->blocks.size() && fInfo->blocks[blockNum]);
        return fInfo->blocks[blockNum];
    }
    
    void FileMgr::putBlock(int fileId, const mapd_size_t blockNum, mapd_addr_t buf) {
        putBlock(getFile(fileId), blockNum, buf);
    }
    
    void FileMgr::putBlock(FileInfo *fInfo, const mapd_size_t blockNum, mapd_addr_t buf) {
        assert(buf);
        openFile(fInfo); // open the file if it is not open already
        
        // write the block to the file
        size_t wrote = writeBlock(fInfo->f, fInfo->blockSize, blockNum, buf);
        assert(wrote == fInfo->blockSize);
    }
    
    void FileMgr::clearBlock(const int fileId, const mapd_size_t blockNum) {
        clearBlock(FileMgr::getFile(fileId), blockNum);
    }
    
    void FileMgr::clearBlock(FileInfo *fInfo, const mapd_size_t blockNum) {
        getBlock(fInfo, blockNum)->used = 0;
    }
    
    void FileMgr::freeBlock(const int fileId, const mapd_size_t blockNum) {
        freeBlock(getFile(fileId), blockNum);
    }
    
    void FileMgr::freeBlock(FileInfo *fInfo, const mapd_size_t blockNum) {
        clearBlock(fInfo, blockNum);
        fInfo->freeBlocks.insert(blockNum);
    }
    
    Chunk* FileMgr::getChunkPtr(const ChunkKey &key) {
        auto chunkIt = chunkIndex_.find(key);
        if (chunkIt != chunkIndex_.end())
            return &chunkIt->second;
        return nullptr;
    }
    
    void FileMgr::copyChunkToBuffer(const ChunkKey &key, mapd_addr_t buf) {
        assert(buf);
        
        // find chunk
        auto it = chunkIndex_.find(key);
        if (it == chunkIndex_.end())
            throw std::runtime_error("Chunk does not exist.");
        
        // copy contents of chunk to buf
        Chunk &c = it->second;
        for (int i = 0; i < c.size(); ++i) {
            
            // get most recent address of current block
            Block *blk = c[i]->current();
            
            // obtain a reference to the file of the block address
            int fileId = blk->fileId;
            FileInfo *fInfo = getFile(fileId);
            if (!fInfo)
                throw std::runtime_error("Unable to obtain file for the requested block.");
            
            // open the file if it is not open already
            openFile(fInfo);
            
            // read block from file into buf
            mapd_size_t used = c[i]->current()->used;
            read(fInfo->f, blk->blockNum * c[i]->blockSize, used, buf);
            buf += used;
        }
    }
    
    void FileMgr::getChunkSize(const ChunkKey &key, mapd_size_t *nblocks, mapd_size_t *size) {
        assert(size || nblocks); // at least one of these should be not NULL
        
        // find the Chunk
        auto it = chunkIndex_.find(key);
        if (it == chunkIndex_.end())
            throw std::runtime_error("Chunk does not exist.");
        Chunk &c = it->second;
        
        // check if chunk has no blocks
        if (nblocks != nullptr) {
            *nblocks = c.size();
            if (*nblocks < 1) {
                if (size != nullptr)
                    *size = 0;
            }
        }
        if (size != nullptr) { // Compute size based on block sizes
            *size = 0;
            for (int i = 0; i < c.size(); ++i)
                *size += c[i]->blockSize;
        }
    }
    
    void FileMgr::getChunkActualSize(const ChunkKey &key, mapd_size_t *size) {
        assert(size);
        
        // find the Chunk
        auto it = chunkIndex_.find(key);
        if (it == chunkIndex_.end())
            throw std::runtime_error("Chunk does not exist.");
        
        // Compute size based on actual bytes used in block
        Chunk &c = it->second;
        for (int i = 0; i < c.size(); ++i)
            *size += c[i]->current()->used;
    }
    
    void FileMgr::copyBufferToChunk(const ChunkKey &key, mapd_size_t size, mapd_addr_t buf) {
        assert(buf);
        
        // ensure Chunk exists
        Chunk* c;
        if ((c = getChunkPtr(key)) == nullptr)
            throw std::runtime_error("Chunk does not exist.");
        
        mapd_size_t blockSize = chunkBlockSize_[key]; // the block size for the Chunk
        mapd_size_t nblocks = (size + blockSize - 1) / blockSize; // number of blocks to copy
        mapd_size_t blockCount = 0; // blockCount: number of blocks written so
        auto fileIt = fileIndex_.lower_bound(blockSize); // file iterator over target block size
        
        // Traverse the MultiBlock* objects of the Chunk, writing blockSize bytes to a new
        // version of each affected logical block of the Chunk
        for (int i = 0; i < c->size(); ++i) {
            
            // find a suitable file (i.e., having the desired block size)
            FileInfo* fInfo = nullptr;
            for (; fileIt != fileIndex_.end(); ++fileIt)
                if (getFile(fileIt->second)->available() > 0)
                    fInfo = getFile(fileIt->second);
            fileIt--; // preserve iterator position
            
            // if no file exists, create a new one
            if (fInfo == nullptr)
                fInfo = createFile(blockSize, MAPD_DEFAULT_N_BLOCKS);
            assert(fInfo->freeBlocks.size() > 0);
            
            // obtain first available free block number, and remove it from free block list
            mapd_size_t freeBlockNum;
            auto freeBlockIt = fInfo->freeBlocks.begin();
            freeBlockNum = *freeBlockIt;
            fInfo->freeBlocks.erase(freeBlockIt);
            
            // Push the obtained block to be used as the new version
            // of the current MultiBlock with the specified epoch
            Block *b = getBlock(fInfo, freeBlockNum);
            (*c)[i]->push(b->fileId, b->blockNum, epoch_);
            
            // Write the contents of the block in buf to the obtained block
            writeBlock(fInfo->f,blockSize,freeBlockNum, buf+blockCount*blockSize);
            
            nblocks--;
            blockCount++;
        }
        
        // Create new MultiBlock objects for the Chunk for remaining unwritten data
        while (nblocks > 0) {
            
            // find a suitable file (i.e., having the desired block size)
            FileInfo* fInfo = NULL;
            for (; fileIt != fileIndex_.end(); ++fileIt)
                if (getFile(fileIt->second)->available() > 0)
                    fInfo = getFile(fileIt->second);
            fileIt--; // preserve iterator position
            
            // if no file exists, create a new one
            if (fInfo == nullptr)
                fInfo = createFile(blockSize, MAPD_DEFAULT_N_BLOCKS);
            assert(fInfo->freeBlocks.size() > 0);
            
            // obtain first available free block number, and remove it from free block list
            mapd_size_t freeBlockNum;
            auto freeBlockIt = fInfo->freeBlocks.begin();
            freeBlockNum = *freeBlockIt;
            fInfo->freeBlocks.erase(freeBlockIt);
            
            // push the new MultiBlock into the Chunk
            MultiBlock* mb = new MultiBlock(fInfo->blockSize);
            Block *b = getBlock(fInfo, freeBlockNum);
            mb->push(b->fileId, b->blockNum, epoch_);
            c->push_back(mb);
            
            // Write the contents of the block in buf to the obtained block
            writeBlock(fInfo->f,blockSize,freeBlockNum, buf+blockCount*blockSize);
            
            nblocks--;
            blockCount++;
            
        } // while (nblocks > 0)
    }

    /// lazy version
    void FileMgr::createChunk(const ChunkKey &key, const mapd_size_t blockSize) {
        assert(blockSize > 0);
        
        // check if Chunk already exists
        auto chunkIt = chunkIndex_.find(key);
        if (chunkIt != chunkIndex_.end())
            throw std::runtime_error("Chunk already exists");
        
        // create new entry in local chunk index
        chunkIndex_.insert(std::pair<ChunkKey, Chunk>(key, Chunk()));
        chunkBlockSize_.insert(std::pair<ChunkKey, mapd_size_t>(key, blockSize));
    }
    
    /// copies the contents of buf into the new Chunk, or if buf is NULL it simply reserves the space
    void FileMgr::createChunk(const ChunkKey &key, const mapd_size_t blockSize, const mapd_size_t size, mapd_addr_t buf) {
        assert(blockSize > 0 && size > 0 && buf);
        
        // creates the Chunk if it doesn't already exist
        createChunk(key, blockSize);
        
        // Calling copyBufferToChunk with a NULL buffer will reserve
        // empty blocks for the Chunk
        copyBufferToChunk(key, size, buf);
    }

    void FileMgr::freeMultiBlock(MultiBlock* mb) {
        while (mb->blkVersions.size() > 0) {
            //get fileInfo of each block
            Block &blk = *mb->blkVersions.front();
            FileInfo *fInfo = getFile(blk.fileId);
            
            // expression refers to file offset of most recent block in MultiBlock
            fInfo->freeBlocks.insert(blk.blockNum);
            // delete the front block
            mb->pop();
        }
        //now, delete the whole multiblock
        delete mb;
    }
    
    void FileMgr::deleteChunk(const ChunkKey &key) {
        Chunk* c = NULL;
        
        // ensure the Chunk exists
        if ((c = getChunkPtr(key)) == NULL)
            throw std::runtime_error("Chunk does not exist.");
        
        // While there are still multiblocks in the chunk, pop the back and free it
        while (c->size() > 0) {
            freeMultiBlock(c->back());
            c->pop_back();
        }
        
        // Remove Chunk from ChunkIndex. Return failure if it does not remove exactly one chunk.
        if (chunkIndex_.erase(key) != 1)
            throw std::runtime_error("Multiple Chunks deleted where there should only have been one.");
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
            f = open(file_ids.back());
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
        
        // READ METADATA: chunk_multiblock and multiblock
        
        // Overview: First, obtain all the unique ChunkKey values. Then, for each ChunkKey,
        // obtain a list of multiblock_id values. Then, for each multiblock_id value, read
        // in the block information for the blocks composing the MultiBlock.
        
        // obtain the unique ChunkKey values from chunk_multiblock and store them in chunkkeys
        std::vector<std::vector<int>> chunkkeys;
        q1 = "select array_to_string(chunkkey, ',') from chunk_multiblock group by chunkkey";
        status = pgConnector_.query(q1);
        assert(status == MAPD_SUCCESS);
        numRows = pgConnector_.getNumRows();
        for (int i = 0; i < numRows; ++i) {
            // each key is initially read in as a string
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
        
        // for each ChunkKey in chunkkeys, create an entry in chunkIndex_, then obtain
        // its list of multiblock_id, then populate the Chunk with its blocks, obtaining
        // the Block metadata from the multiblock table
        for (int i = 0; i < chunkkeys.size(); ++i) {
            // string representation of the ChunkKey used by the queries
            std::string strChunkKey = "'{";
            for (int j = 0; j < chunkkeys[i].size(); ++j) {
                strChunkKey += std::to_string(chunkkeys[i][j]);
                if (j+1 < chunkkeys[i].size())
                    strChunkKey += ",";
            }
            strChunkKey += "}'";
            
            // build query to obtain multiblock metadata for current Chunk
            q1 = "select multiblock_id, block_id, epoch, file_id, block_num from multiblock where multiblock_id in (select multiblock_id from chunk_multiblock where chunkkey = '{";
            for (int j = 0; j < chunkkeys[i].size(); ++j) {
                q1 += std::to_string(chunkkeys[i][j]);
                if (j+1 < chunkkeys[i].size())
                    q1 += ",";
            }
            q1 += "}') order by block_id asc;";
            printf("q1 = %s\n", q1.c_str());
            
            // execute query
            status = pgConnector_.query(q1);
            assert(status == MAPD_SUCCESS);
            numRows = pgConnector_.getNumRows();

            // push the blocks into the current MultiBlock
            Chunk c(numRows);
            
            for (int i = 0; i < numRows; ++i) {
                // get block metadata
                int block_id = pgConnector_.getData<int>(i,0);
                int epoch = pgConnector_.getData<int>(i,1);
                int file_id = pgConnector_.getData<int>(i,2);
                int block_num = pgConnector_.getData<int>(i,3);
                
                // obtain the next Block pointer for the Chunk using the obtained metadata
                Block *b = files_[file_id]->blocks[block_num];
                
                // update epoch and insert the Block as a version of the current MultiBlock
                c[block_id]->epochs[block_id] = epoch;
                c[block_id]->blkVersions.push_back(b);
            }
            
            // insert entry into chunkIndex_ for this Chunk with the number of multiblocks (numRows)
            chunkIndex_.insert(std::pair<ChunkKey, Chunk>(chunkkeys[i], c));
        }
    }
    
    void FileMgr::writeState() {
        std::string q1, q2, q3;
        mapd_err_t status;

        // CLEAR THE EXISTING METADATA TABLES
        clearState();
        
        // WRITE METADATA: FileInfo (save the state of "vector<FileInfo> files_")
        for (int i = 0; i < files_.size(); ++i) {
            FileInfo *fInfo = files_[i];

            q1 = "insert into fileinfo(file_id, block_size, nblocks) values (" + std::to_string(fInfo->fileId) + ", " + std::to_string(fInfo->blockSize) + ", " + std::to_string(fInfo->nblocks) + ");";
            
            status = pgConnector_.query(q1);
            assert(status == MAPD_SUCCESS);
            
            // WRITE METADATA: FileInfo_blocks (save the state for each of the file's blocks)
            assert(fInfo->nblocks == fInfo->blocks.size());
            for (int j = 0; j < fInfo->nblocks; ++j) {
                Block *b = fInfo->blocks[j];
                q1 = "insert into fileinfo_blocks(file_id, block_num, used) values (" + std::to_string(b->fileId) + ", " + std::to_string(b->blockNum) + ", " + std::to_string(b->used) + ");";
                status = pgConnector_.query(q1);
                assert(status == MAPD_SUCCESS);
            }
        }
        
        // WRITE METADATA: chunk_multiblock and multiblock
        for (auto chunkIt = chunkIndex_.begin(); chunkIt != chunkIndex_.end(); ++chunkIt) {
            ChunkKey key = chunkIt->first;
            Chunk *c = &(chunkIt->second);
            
            int multiblock_id = 0;
            for (auto mbIt = c->begin(); mbIt != c->end(); ++mbIt, ++multiblock_id) {

                q1 = "insert into chunk_multiblock(chunkkey, multiblock_id) values('{";
                for (int i = 0; i < key.size(); ++i) {
                    q1 += std::to_string(key[i]);
                    if (i+1 < key.size())
                        q1 += ",";
                }
                q1 += "}', " + std::to_string(multiblock_id) + ");";
                printf("query = %s\n", q1.c_str());
                status = pgConnector_.query(q1);
                assert(status == MAPD_SUCCESS);
                
                MultiBlock *mb = *mbIt;
                int block_id = 0;
                for (auto blockIt = mb->blkVersions.begin(); blockIt != mb->blkVersions.end(); ++blockIt) {
                    Block *b = *blockIt;
                    
                    q1 = "insert into multiblock(multiblock_id, block_id, epoch, file_id, block_num) values(";
                    q1 += std::to_string(multiblock_id) + ", ";
                    q1 += std::to_string(block_id) + ", ";
                    q1 += std::to_string(mb->epochs[block_id]) + ", ";
                    q1 += std::to_string(b->fileId) + ", ";
                    q1 += std::to_string(b->blockNum);
                    q1 += ");";
                    printf("query = %s\n", q1.c_str());
                }
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
