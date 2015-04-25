#include "FileInfo.h"
#include "Page.h"
#include "File.h"
#include <glog/logging.h>
#include <iostream>

#include <utility>
using namespace std;

#define CHECK_RET(x) CHECK_GE(x, 0)

namespace File_Namespace {

    FileInfo::FileInfo(const int fileId, FILE *f, const size_t pageSize, size_t numPages, bool init)
    : fileId(fileId), f(f), pageSize(pageSize), numPages(numPages) {
        if (init) {
            initNewFile();
        }
    }

    FileInfo::~FileInfo() {
        // close file, if applicable
        if (f)
            close(f);
    }

    void FileInfo::initNewFile() {
        // initialize pages and free page list
        // Also zeroes out first four bytes of every header
        
        int headerSize = 0;
        int8_t * headerSizePtr = (int8_t *)(&headerSize);
        for (size_t pageId = 0; pageId < numPages; ++pageId) {
            write(f,pageId*pageSize,sizeof(int),headerSizePtr);
            freePages.insert(pageId);
        }
    }

    void FileInfo::openExistingFile(std::vector<HeaderInfo> &headerVec, const int fileMgrEpoch) {
        //HeaderInfo is defined in Page.h
        for (size_t pageNum = 0; pageNum < numPages; ++pageNum) {
            int headerSize;
            fseek(f, pageNum*pageSize, SEEK_SET);
            CHECK_RET(fread((int8_t *)(&headerSize),sizeof(int),1,f));
            if (headerSize != 0) {
                // headerSize doesn't include headerSize itself
                // We're tying ourself to headers of ints here
                size_t numHeaderElems = headerSize/sizeof(int);
                assert(numHeaderElems >= 2);
                // Last two elements of header are always PageId and Version
                // epoch - these are not in the chunk key so seperate them
                std::vector<int> chunkKey (numHeaderElems - 2); 
                int pageId;
                int versionEpoch;
                //size_t chunkSize;
                // We don't want to read headerSize in our header - so start
                // reading 4 bytes past it
                CHECK_RET(fread((int8_t *)(&chunkKey[0]),headerSize - 2*sizeof(int),1,f));
                //cout << "Chunk key: " << chunkKey[0] << endl;
                CHECK_RET(fread((int8_t *)(&pageId),sizeof(int),1,f));
                //cout << "Page id: " << pageId << endl;
                CHECK_RET(fread((int8_t *)(&versionEpoch),sizeof(int),1,f));
                //read(f,pageNum*pageSize+sizeof(int),headerSize-2*sizeof(int),(int8_t *)(&chunkKey[0]));
                //read(f,pageNum*pageSize+sizeof(int) + headerSize - 2*sizeof(int),sizeof(int),(int8_t *)(&pageId));
                //read(f,pageNum*pageSize+sizeof(int) + headerSize - sizeof(int),sizeof(int),(int8_t *)(&versionEpoch));
                //read(f,pageNum*pageSize+sizeof(int) + headerSize - sizeof(size_t),sizeof(size_t),(int8_t *)(&chunkSize));

                /* Check if version epoch is equal to 
                 * or greater (note: should never be greater)
                 * than FileMgr epoch_ - this means that this
                 * page wasn't checkpointed and thus we should
                 * not use it 
                 */
                if (versionEpoch >= fileMgrEpoch) {
                    // First write 0 to first four bytes of
                    // header to mark as free
                    headerSize = 0;
                    write(f,pageNum*pageSize,sizeof(int),(int8_t *)&headerSize);
                    // Now add page to free list
                    freePages.insert(pageNum);
                    //std::cout << "Not checkpointed" << std::endl;

                }
                else { // page was checkpointed properly
                    Page page(fileId,pageNum);
                    headerVec.push_back(HeaderInfo(chunkKey,pageId,versionEpoch,page));
                    //std::cout << "Inserted into headerVec" << std::endl;
                }
            }
            else { // no header for this page - insert into free list
                freePages.insert(pageNum);
            }
        }
    }

    void FileInfo::freePage(int pageId) {
        int zeroVal = 0;
        int8_t * zeroAddr = reinterpret_cast<int8_t *> (&zeroVal);
        write(f,pageId*pageSize,sizeof(int),zeroAddr);
        std::lock_guard < std::mutex > lock (freePagesMutex_);
        freePages.insert(pageId);
    }

    int FileInfo::getFreePage() {
        // returns -1 if there is no free page
        std::lock_guard < std::mutex > lock (freePagesMutex_);
        if (freePages.size() == 0)  {
            return - 1;
        }
        auto pageIt = freePages.begin();
        int pageNum = *pageIt;
        freePages.erase(pageIt);
        return pageNum;
    }
        
        
        


    
    
    void FileInfo::print(bool pagesummary) {
        std::cout << "File: " << fileId << std::endl;
        std::cout << "Size: " << size() << std::endl;
        std::cout << "Used: " << used() << std::endl;
        std::cout << "Free: " << available() << std::endl;
        if (!pagesummary)
            return;
        
        //for (size_t i = 0; i < pages.size(); ++i) {
        //    // @todo page summary
        //}
    }
} // File_Namespace
