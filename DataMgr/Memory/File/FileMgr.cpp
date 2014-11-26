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
        for (int i = 0; i < files_.size(); ++i) {
            delete files_[i];
        }
    }

    AbstractDatum* FileMgr::createChunk(const ChunkKey &key, mapd_size_t pageSize) {
        // we will do this lazily and not allocate space for the Chunk (i.e.
        // FileBuffer yet)
        if (chunkIndex_.find(key) != chunkIndex_.end()) {
            Chunk *chunk = new Chunk (this,pageSize);
            //chunkIndex_[key] = std::move(Chunk(this,pageSize)); // should avoid copy?
            //chunkIndex_[key] = std::move(chunk); // should avoid copy?
            chunkIndex_[key] = chunk;
            return (chunkIndex_[key]);
            //return 0;
            //Chunk chunk(pageSize,this);
            //chunkIndex_[key] = chunk;
        }
        else {
            throw std::runtime_error("Chunk already exists.");
            return 0;
        }
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
        return chunkIt->second;
    }

    AbstractDatum* FileMgr::putChunk(const ChunkKey &key, AbstractDatum *datum) {
        // obtain a pointer to the Chunk
        auto chunkIt = chunkIndex_.find(key);
        AbstractDatum *chunk;
        if (chunkIt == chunkIndex_.end()) {
            chunk = createChunk(key,datum->pageSize());
        }
        else {
            chunk = chunkIt->second;
        }
        // write the datum's data to the Chunk
        chunk->write((mapd_addr_t)datum->getMemoryPtr(), 0, datum->pageSize() * datum->pageCount());
        return chunk;
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

    Page FileMgr::requestFreePage(mapd_size_t pageSize) {
        auto candidateFiles = fileIndex_.equal_range(pageSize);
        FileInfo * freeFile = 0;
        for (auto fileIt = candidateFiles.first; fileIt != candidateFiles.second; ++fileIt) {
            mapd_size_t fileFreePages = files_[fileIt->second]->numFreePages();
            freeFile = files_[fileIt->second];
        }
        if (freeFile == nullptr) { // didn't find free file and need to create one
            freeFile = createFile(pageSize, MAPD_DEFAULT_N_PAGES);
        }
        assert (freeFile != nullptr); // should have file by now
        auto pageNumIt = freeFile -> freePages.begin();
        mapd_size_t pageNum = *pageNumIt;
        freeFile -> freePages.erase(pageNum);
        Page page (freeFile->fileId,pageNum);
        return page;
    }

    void FileMgr::requestFreePages(mapd_size_t numPagesRequested, mapd_size_t pageSize, std::vector<Page> &pages) {
        mapd_size_t numFreePagesAvailable = 0;
        std::vector<FileInfo*> fileList;
        
        // determine if there are enough free pages available
        auto candidateFiles = fileIndex_.equal_range(pageSize);
        for (auto fileIt = candidateFiles.first; fileIt != candidateFiles.second; ++fileIt) {
            mapd_size_t fileFreePages = files_[fileIt->second]->numFreePages();
            if (fileFreePages > 0) {
                fileList.push_back(files_[fileIt->second]);
                numFreePagesAvailable += fileFreePages;
            }
        }
        // create new file(s) if there are not enough free pages
        mapd_size_t numPagesToCreate = numPagesRequested - numFreePagesAvailable;
        mapd_size_t numFilesToCreate = (numPagesToCreate + MAPD_DEFAULT_N_PAGES - 1) / MAPD_DEFAULT_N_PAGES;
        for (; numFilesToCreate > 0; --numFilesToCreate) {
            FileInfo *fInfo = createFile(pageSize, MAPD_DEFAULT_N_PAGES);
            if (fInfo == nullptr)
                throw std::runtime_error("Unable to create file for free pages.");
            fileList.push_back(fInfo);
        }

        mapd_size_t numPagesRemaining = numPagesRequested;
        for (size_t i = 0; i < fileList.size() && numPagesRemaining > 0; ++i) {
            FileInfo *fInfo = fileList[i];
            //assert(fInfo->freePages.size() == MAPD_DEFAULT_N_PAGES); // error?? this will only be true if the file is newly created

            for (auto pageNumIt = fInfo->freePages.begin(); pageNumIt != fInfo->freePages.end() &&
                numPagesRemaining > 0; --numPagesRemaining) {

                mapd_size_t pageNum = *pageNumIt;
                fInfo->freePages.erase(pageNumIt++);
                pages.push_back(Page(fInfo->fileId, pageNum));
                //changed from Steve's code - make sure pasts test
            }
        }
        assert(pages.size() == numPagesRequested);
    }

    FileInfo* FileMgr::createFile(const mapd_size_t pageSize, const mapd_size_t numPages) {
        // check arguments
        if (pageSize == 0 || numPages == 0)
            throw std::invalid_argument("pageSize and numPages must be greater than 0.");
        
        // create the new file
        FILE *f = create(nextFileId_, pageSize, numPages); //TM: not sure if I like naming scheme here - should be in separate namespace?
        if (f == nullptr)
            throw std::runtime_error("Unable to create the new file.");
        
        // instantiate a new FileInfo for the newly created file
        int fileId = nextFileId_++;
        FileInfo *fInfo = new FileInfo(fileId, f, pageSize, numPages);
        assert(fInfo);
        
        // update file manager data structures
        files_.push_back(fInfo);
        fileIndex_.insert(std::pair<mapd_size_t, int>(pageSize, fileId));
        
        assert(files_.back() == fInfo); // postcondition
        return fInfo;
    }

    FILE * FileMgr::getFileForFileId(const int fileId) {
        assert (fileId >= 0 && fileId < nextFileId_);
        return files_[fileId] -> f;
    }



} // File_Namespace
