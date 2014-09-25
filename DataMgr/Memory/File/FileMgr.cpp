/**
 * @file        FileMgr.h
 * @author      Steven Stewart <steve@map-d.com>
 */
#include <string>
#include "FileMgr.h"
#include "File.h"
#include "../../../Shared/global.h"

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
    
    FileMgr::FileMgr(std::string basePath) : basePath_(basePath), pgConnector_("mapd", "mapd"), nextFileId_(0), epoch_(0) {
        // NOP
    }
    
    FileMgr::~FileMgr() {
        // free memory used by FileInfo objects
        for (int i = 0; i < files_.size(); ++i)
            delete files_[i];
    }
    
    void FileMgr::createChunk(const ChunkKey &key, mapd_size_t pageSize) {
        if (chunkBlockSize_.find(key) != chunkBlockSize_.end())
            throw std::runtime_error("Chunk already exists.");
        chunkBlockSize_[key] = pageSize;
    }
    
    void FileMgr::deleteChunk(const ChunkKey &key) {
        // ensure the Chunk exists
        if (chunkBlockSize_.find(key) == chunkBlockSize_.end()) {
            assert(chunkIndex_.find(key) == chunkIndex_.end()); // there should not be a chunkIndex_ entry
            throw std::runtime_error("Chunk does not exist.");
        }
        
        // check if the Chunk contents are stored in the file system
        auto chunkIt = chunkIndex_.find(key);
        if (chunkIt != chunkIndex_.end()) {
            
        }
        
        // remove ChunkKey from chunkIndex_; throw an exception if it does not remove exactly one chunk
        if (chunkIndex_.erase(key) != 1)
            throw std::runtime_error("Multiple Chunks deleted where there should only have been one.");
        
        // remove ChunkKey from chunkBlockSize_ map
        chunkBlockSize_.erase(key);
    }
    
    AbstractDatum* FileMgr::getChunk(ChunkKey &key) {
        auto chunkIt = chunkIndex_.find(key);
        if (chunkIt == chunkIndex_.end())
            throw std::runtime_error("Chunk does not exist.");
        return &chunkIt->second;
    }
    
    /// "Chunk" is an alias for "FileBuffer"
    AbstractDatum* FileMgr::putChunk(const ChunkKey &key, AbstractDatum *d) {
        // obtain a pointer to the Chunk
        auto chunkIt = chunkIndex_.find(key);
        if (chunkIt == chunkIndex_.end())
            throw std::runtime_error("Chunk does not exist.");
        Chunk *c = &chunkIt->second;
        
        // write the d's data to the Chunk
        c->write((mapd_addr_t)d->getMemoryPtr(), 0, d->pageSize() * d->pageCount());
        
        return c;
    }
    
    AbstractDatum* FileMgr::createDatum(mapd_size_t pageSize, mapd_size_t nbytes) {
        return nullptr;
    }
    
    void FileMgr::deleteDatum(AbstractDatum *d) {

    }
    
    AbstractDatum* FileMgr::putDatum(AbstractDatum *d) {
        return nullptr;
    }
    
    void FileMgr::requestFreeBlocks(mapd_size_t nblocks, mapd_size_t blockSize, std::vector<Block> &blocks) {
        mapd_size_t numFreeBlocksAvailable = 0;
        std::vector<FileInfo*> fileList;
        
        // determine if there are enough free blocks available
        auto ret = fileIndex_.equal_range(blockSize);
        for (auto fileIt = ret.first; fileIt != ret.second; ++fileIt) {
            mapd_size_t tmp = files_[fileIt->second]->numFreeBlocks();
            if (tmp > 0)
                fileList.push_back(files_[fileIt->second]);
            numFreeBlocksAvailable += files_[fileIt->second]->numFreeBlocks();
        }
        
        // create new file(s) if there are not enough free blocks
        mapd_size_t numBlocksToCreate = nblocks - numFreeBlocksAvailable;
        mapd_size_t numFilesToCreate = (numBlocksToCreate + MAPD_DEFAULT_N_BLOCKS - 1) / MAPD_DEFAULT_N_BLOCKS;
        for (; numFilesToCreate > 0; --numFilesToCreate) {
            FileInfo *fInfo = createFile(blockSize, MAPD_DEFAULT_N_BLOCKS);
            if (fInfo == nullptr)
                throw std::runtime_error("Unable to create file for free blocks.");
            fileList.push_back(fInfo);
        }
        
        // Traverse the file list, adding the free blocks to the blocks vector, and erasing
        // the acquired blocks from their respective free block lists
        mapd_size_t numBlocksRemaining = nblocks;
        for (size_t i = 0; i < fileList.size() && numBlocksRemaining > 0; ++i) {
            FileInfo *fInfo = fileList[i];
            assert(fInfo->freeBlocks.size() == MAPD_DEFAULT_N_BLOCKS);

            for (auto blockNumIt = fInfo->freeBlocks.begin(); blockNumIt != fInfo->freeBlocks.end() &&
                numBlocksRemaining > 0; --numBlocksRemaining) {

                mapd_size_t blockNum = *blockNumIt;
                ++blockNumIt; // advance to next element before erasing
                
                blocks.push_back(Block(fInfo->fileId, blockNum));
                fInfo->freeBlocks.erase(blockNum);
                
            }
        }
        assert(blocks.size() == nblocks);
        
        /*printf("blocks.size() = %lu\n", blocks.size());
        for (int i = 0; i < blocks.size(); ++i)
            printf("block: fileId=%d blockNum=%lu\n", blocks[i].fileId, blocks[i].blockNum);*/
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
    
}