/**
 * @file        FileMgr.h
 * @author      Steven Stewart <steve@map-d.com>
 * @author      Todd Mostak <todd@map-d.com>
 */

#include "FileMgr.h"
#include "File.h"
#include "../../../Shared/global.h"
#include <string>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include <vector>
#include <utility>
#include <algorithm>
#include <unistd.h>

#define EPOCH_FILENAME "epoch"

namespace File_Namespace {

    bool headerCompare(const HeaderInfo &firstElem, const HeaderInfo &secondElem) {
        // HeaderInfo.first is a pair of Chunk key with a vector containing
        // pageId and version
        if(firstElem.chunkKey != secondElem.chunkKey) 
            return firstElem.chunkKey < secondElem.chunkKey;
        if (firstElem.pageId != secondElem.pageId)
            return firstElem.pageId < secondElem.pageId;
        return firstElem.versionEpoch < secondElem.versionEpoch;

        /*
        if (firstElem.first.first != secondElem.first.first)
            return firstElem.first.first < secondElem.first.first;
        return firstElem.first.second < secondElem.first.second;
        */
    }


    FileMgr::FileMgr(std::string basePath) : basePath_(basePath), nextFileId_(0), epoch_(0) {
        init();
    }

    FileMgr::~FileMgr() {
        checkpoint();
        // free memory used by FileInfo objects
        for (auto chunkIt = chunkIndex_.begin(); chunkIt != chunkIndex_.end(); ++chunkIt) {
            delete chunkIt -> second;
        }
        for (int i = 0; i < files_.size(); ++i) {
            delete files_[i];
        }
    }


    void FileMgr::init() {
        boost::filesystem::path path (basePath_);
        if (basePath_.size() > 0 && basePath_[basePath_.size()-1] != '/')
            basePath_.push_back('/');
        if (boost::filesystem::exists(path)) {
            if (!boost::filesystem::is_directory(path))
                throw std::runtime_error("Specified path is not a directory.");
            std::cout << basePath_ << " exists." << std::endl;
            openEpochFile(EPOCH_FILENAME);

            boost::filesystem::directory_iterator endItr; // default construction yields past-the-end
            int maxFileId = -1;
            std::vector <HeaderInfo> headerVec;
            for (boost::filesystem::directory_iterator fileIt (path); fileIt != endItr; ++fileIt) {
                if (boost::filesystem::is_regular_file(fileIt ->status())) {
                    //note that boost::filesystem leaves preceding dot on
                    //extension - hence MAPD_FILE_EXT is ".mapd"
                    std::string extension (fileIt ->path().extension().string());
                    
                    if (extension == MAPD_FILE_EXT) { 
                        std::string fileStem(fileIt -> path().stem().string());
                        // remove trailing dot if any
                        if (fileStem.size() > 0 && fileStem.back() == '.') {
                            fileStem = fileStem.substr(0,fileStem.size()-1);
                        }
                        size_t dotPos =  fileStem.find_last_of("."); // should only be one
                        if (dotPos == std::string::npos) {
                            throw std::runtime_error("Filename does not carry page size information");
                        }
                        int fileId = boost::lexical_cast<int>(fileStem.substr(0,dotPos));
                        if (fileId > maxFileId) {
                            maxFileId = fileId;
                        }
                        mapd_size_t pageSize = boost::lexical_cast<mapd_size_t>(fileStem.substr(dotPos+1,fileStem.size()));
                        std::string filePath(fileIt ->path().string());
                        size_t fileSize = boost::filesystem::file_size(filePath);
                        assert (fileSize % pageSize == 0); // should be no partial pages
                        mapd_size_t numPages = fileSize / pageSize;

                        std::cout << "File id: " << fileId << " Page size: " << pageSize << " Num pages: " << numPages << std::endl;
                        openExistingFile(filePath,fileId,pageSize,numPages,headerVec);
                    }
                }
            }

            /* Sort headerVec so that all HeaderInfos
             * from a chunk will be grouped together 
             * and in order of increasing PageId
             * - Version Epoch */

            std::sort(headerVec.begin(),headerVec.end(),headerCompare);

            /* Goal of next section is to find sequences in the
             * sorted headerVec of the same ChunkId, which we
             * can then initiate a FileBuffer with */

            std::cout << "Header vec size: " << headerVec.size() << std::endl;
            if (headerVec.size() > 0) {
                ChunkKey lastChunkKey = headerVec.begin() -> chunkKey;
                auto startIt = headerVec.begin();

                for (auto headerIt = headerVec.begin() + 1 ; headerIt != headerVec.end(); ++headerIt) {
                    for (auto chunkIt = headerIt -> chunkKey.begin(); chunkIt != headerIt -> chunkKey.end(); ++chunkIt) {
                        std::cout << *chunkIt << " ";
                    }
                    std::cout << " -> " << headerIt -> pageId << "," << headerIt -> versionEpoch << std::endl;
                    if (headerIt -> chunkKey != lastChunkKey) {
                        std::cout << "New chunkkey" << std::endl;
                        
                        mapd_size_t pageSize = files_[startIt -> page.fileId] -> pageSize;
                        chunkIndex_[lastChunkKey] = new Chunk (this,pageSize,lastChunkKey,startIt,headerIt);
                        lastChunkKey = headerIt -> chunkKey;
                        startIt = headerIt;
                    }
                }
                // now need to insert last Chunk
                mapd_size_t pageSize = files_[startIt -> page.fileId] -> pageSize;
                chunkIndex_[lastChunkKey] = new Chunk (this,pageSize,lastChunkKey,startIt,headerVec.end());

            }
            nextFileId_ = maxFileId + 1;
        }
        else { // data directory does not exist
            std::cout << basePath_ << " does not exist. Creating" << std::endl;
            if (!boost::filesystem::create_directory(path)) {
                throw std::runtime_error("Could not create data directory");
            }
            std::cout << basePath_ << " created." << std::endl;
            // now create epoch file
            createEpochFile(EPOCH_FILENAME);
        }

    }

    void FileMgr::createEpochFile(const std::string &epochFileName) {
        std::string epochFilePath(basePath_ + epochFileName);
        if (boost::filesystem::exists(epochFilePath)) {
            throw std::runtime_error("Epoch file already exists");
        }
        epochFile_ = create(epochFilePath,sizeof(int));
        // Write out current epoch to file - which if this
        // function is being called should be 0
        write(epochFile_,0,sizeof(int),(mapd_addr_t)&epoch_);
        epoch_++;
    }

    void FileMgr::openEpochFile(const std::string &epochFileName) {
        std::string epochFilePath(basePath_ + epochFileName);
        if (!boost::filesystem::exists(epochFilePath)) {
            throw std::runtime_error("Epoch file does not exist");
        }
        if (!boost::filesystem::is_regular_file(epochFilePath)) {
            throw std::runtime_error("Epoch file is not a regular file");
        }
        if (boost::filesystem::file_size(epochFilePath) < 4) {
            throw std::runtime_error("Epoch file is not sized properly");
        }
        epochFile_ = open(epochFilePath);
        read(epochFile_,0,sizeof(int),(mapd_addr_t)&epoch_);
        std::cout << "Epoch after open file: " << epoch_ << std::endl;
        epoch_++; // we are in new epoch from last checkpoint
    }

    void FileMgr::writeAndSyncEpochToDisk() {
        write(epochFile_,0,sizeof(int),(mapd_addr_t)&epoch_);
        int status = fsync(fileno(epochFile_)); // gets file descriptor for epoch file and then uses it to fsync
        if (status != 0) {
            throw std::runtime_error("Could not sync epoch file to disk");
        }
        
        ++epoch_;
    }

    void FileMgr::checkpoint() {
        std::cout << "Checkpointing " << epoch_ <<  std::endl;
        for (auto fileIt = files_.begin(); fileIt != files_.end(); ++fileIt) {
            int status = (*fileIt) -> syncToDisk();
            if (status != 0)
                throw std::runtime_error("Could not sync file to disk");
        }
        writeAndSyncEpochToDisk();
    }


    AbstractDatum* FileMgr::createChunk(const ChunkKey &key, const mapd_size_t pageSize, const mapd_size_t numBytes) {
        // we will do this lazily and not allocate space for the Chunk (i.e.
        // FileBuffer yet)
        if (chunkIndex_.find(key) != chunkIndex_.end()) {
            throw std::runtime_error("Chunk already exists.");
        }
        chunkIndex_[key] = new Chunk (this,pageSize,key,numBytes);
        return (chunkIndex_[key]);
    }

    void FileMgr::deleteChunk(const ChunkKey &key) {
        auto chunkIt = chunkIndex_.find(key);
        // ensure the Chunk exists
        if (chunkIt == chunkIndex_.end()) {
            throw std::runtime_error("Chunk does not exist.");
        }
        chunkIt -> second -> freePages();
        delete chunkIt -> second;
        chunkIndex_.erase(chunkIt);
    }

    AbstractDatum* FileMgr::getChunk(ChunkKey &key) {
        auto chunkIt = chunkIndex_.find(key);
        if (chunkIt == chunkIndex_.end())
            throw std::runtime_error("Chunk does not exist.");
        return chunkIt->second;
    }

    
    //void FileMgr::fetchChunk(const ChunkKey &key, AbstractDatum *destDatum, const mapd_size_t numBytes) {
    //    // reads chunk specified by ChunkKey into AbstractDatum provided by
    //    // destDatum
    //    auto chunkIt = chunkIndex_.find(key);
    //    if (chunkIt == chunkIndex_.end()) 
    //        throw std::runtime_error("Chunk does not exist");
    //    AbstractDatum *chunk = chunkIt -> second;
    //    // ChunkSize is either specified in function call with numBytes or we
    //    // just look at pageSize * numPages in FileBuffer
    //    mapd_size_t chunkSize = numBytes == 0 ? chunk->size() : numBytes;
    //    datum->reserve(chunkSize);
    //    chunk->read(datum->getMemoryPtr(),chunkSize,0);
    //}

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
            if (fileFreePages > 0) {
                freeFile = files_[fileIt->second];
                break;
            }
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

    FileInfo* FileMgr::openExistingFile(const std::string &path, const int fileId, const mapd_size_t pageSize, const mapd_size_t numPages, std::vector<HeaderInfo> &headerVec) {

        FILE *f = open(path);
        FileInfo *fInfo = new FileInfo (fileId, f, pageSize, numPages,false); // false means don't init file
        
        fInfo -> openExistingFile(headerVec,epoch_);
        if (fileId >= files_.size()) {
            files_.resize(fileId+1);
        }
        files_[fileId] = fInfo;
        fileIndex_.insert(std::pair<mapd_size_t, int>(pageSize, fileId));
        return fInfo;
    }

    FileInfo* FileMgr::createFile(const mapd_size_t pageSize, const mapd_size_t numPages) {
        // check arguments
        if (pageSize == 0 || numPages == 0)
            throw std::invalid_argument("pageSize and numPages must be greater than 0.");
        
        // create the new file
        FILE *f = create(basePath_,nextFileId_, pageSize, numPages); //TM: not sure if I like naming scheme here - should be in separate namespace?
        if (f == nullptr)
            throw std::runtime_error("Unable to create the new file.");
        
        // instantiate a new FileInfo for the newly created file
        int fileId = nextFileId_++;
        FileInfo *fInfo = new FileInfo(fileId, f, pageSize, numPages, true); // true means init file
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
