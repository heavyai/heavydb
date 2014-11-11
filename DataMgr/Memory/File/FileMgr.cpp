/**
 * @file        FileMgr.h
 * @author      Steven Stewart <steve@map-d.com>
 * @author      Todd Mostak <todd@map-d.com>
 */

#include <string>
#include "FileMgr.h"
#include "File.h"
#include "../../../Shared/global.h"

namespace File_Namespace {

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
        auto chunkIt = chunkIndex_.find(key);
        // ensure the Chunk exists
        if (chunkIt == chunkIndex_.end()) {
            throw std::runtime_error("Chunk does not exist.");
        }
        // does the below happen automatically
        delete chunkIt -> second;
        chunkIndex_.erase(chunkIt);
    }

    AbstractDatum* FileMgr::getChunk(ChunkKey &key) {
        auto chunkIt = chunkIndex_.find(key);
        if (chunkIt == chunkIndex_.end())
            throw std::runtime_error("Chunk does not exist.");
        return &chunkIt->second;
    }

    AbstractDatum* FileMgr::putChunk(const ChunkKey &key, AbstractDatum *datum) {
        // TM: do we really want to only be able to put to a chunk that already
        // exists?
        
        // obtain a pointer to the Chunk
        auto chunkIt = chunkIndex_.find(key);
        if (chunkIt == chunkIndex_.end())
            throw std::runtime_error("Chunk does not exist.");
        Chunk *chunk = &chunkIt->second;
        // write the d's data to the Chunk
        chunk->write((mapd_addr_t)datum->getMemoryPtr(), 0, datum->pageSize() * datum->pageCount());
        
        return c;
    }

    AbstractDatum* FileMgr::createDatum(mapd_size_t pageSize, mapd_size_t nbytes) {
        throw std::runtime_error("Operation not supported");
    }
    
    void FileMgr::deleteDatum(AbstractDatum *d) {
        throw std::runtime_error("Operation not supported");

    }
    
    AbstractDatum* FileMgr::putDatum(AbstractDatum *d) {
        throw std::runtime_error("Operation not supported");
    }

    void FileMgr::requestFreeBlocks(mapd_size_t numBlocksRequested, mapd_size_t blockSize, std::vector<Block> &blocks) {
        mapd_size_t numFreeBlocksAvailable = 0;
        std::vector<FileInfo*> fileList;
        
        // determine if there are enough free blocks available
        auto candidateFiles = fileIndex_.equal_range(blockSize);
        for (auto fileIt = candidateFiles.first; fileIt != candidateFiles.second; ++fileIt) {
            mapd_size_t fileFreeBlocks = files_[fileIt->second]->numFreeBlocks();
            if (fileFreeBlocks > 0) {
                fileList.push_back(files_[fileIt->second]);
                numFreeBlocksAvailable += fileFreeBlocks;
            }
        }
        // create new file(s) if there are not enough free blocks
        mapd_size_t numBlocksToCreate = numBlocksRequested - numFreeBlocksAvailable;
        mapd_size_t numFilesToCreate = (numBlocksToCreate + MAPD_DEFAULT_N_BLOCKS - 1) / MAPD_DEFAULT_N_BLOCKS;
        for (; numFilesToCreate > 0; --numFilesToCreate) {
            FileInfo *fInfo = createFile(blockSize, MAPD_DEFAULT_N_BLOCKS);
            if (fInfo == nullptr)
                throw std::runtime_error("Unable to create file for free blocks.");
            fileList.push_back(fInfo);
        }

        mapd_size_t numBlocksRemaining = numBlocksRequested;
        for (size_t i = 0; i < fileList.size() && numBlocksRemaining > 0; ++i) {
            FileInfo *fInfo = fileList[i];
            assert(fInfo->freeBlocks.size() == MAPD_DEFAULT_N_BLOCKS);

            for (auto blockNumIt = fInfo->freeBlocks.begin(); blockNumIt != fInfo->freeBlocks.end() &&
                numBlocksRemaining > 0; --numBlocksRemaining) {

                mapd_size_t blockNum = *blockNumIt;
                fInfo->freeBlocks.erase(blockNumIt++);
                blocks.push_back(Block(fInfo->fileId, blockNum));
                //changed from Steve's code - make sure pasts test
            }
        }
        assert(blocks.size() == numBlocksRequested);
    }

    FileInfo* FileMgr::createFile(const mapd_size_t blockSize, const mapd_size_t numBlocks) {
        // check arguments
        if (blockSize == 0 || numBlocks == 0)
            throw std::invalid_argument("blockSize and numBlocks must be greater than 0.");
        
        // create the new file
        FILE *f = create(nextFileId_, blockSize, numBlocks); //TM: not sure if I like naming scheme here - should be in separate namespace?
        if (f == nullptr)
            throw std::runtime_error("Unable to create the new file.");
        
        // instantiate a new FileInfo for the newly created file
        int fileId = nextFileId_++;
        FileInfo *fInfo = new FileInfo(fileId, f, blockSize, numBlocks);
        assert(fInfo);
        
        // update file manager data structures
        files_.push_back(fInfo);
        fileIndex_.insert(std::pair<mapd_size_t, int>(blockSize, fileId));
        
        assert(files_.back() == fInfo); // postcondition
        return fInfo;
    }


} // File_Namespace
