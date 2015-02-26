/**
 * @file        FileBuffer.cpp
 * @author      Steven Stewart <steve@map-d.com>
 * @author      Todd Mostak <todd@map-d.com>
 */

#include "FileBuffer.h"
#include "File.h"
#include "FileMgr.h"
#include <map>

#define METADATA_PAGE_SIZE 4096

using namespace std;

namespace File_Namespace {
    size_t FileBuffer::headerBufferOffset_ = 32;

    FileBuffer::FileBuffer(FileMgr *fm, const size_t pageSize, const ChunkKey &chunkKey, const size_t initialSize) : AbstractBuffer(fm->getDeviceId()), fm_(fm), metadataPages_(METADATA_PAGE_SIZE), pageSize_(pageSize), chunkKey_(chunkKey) {
        // Create a new FileBuffer
        assert(fm_);
        calcHeaderBuffer();
        pageDataSize_ = pageSize_-reservedHeaderSize_;
        //@todo reintroduce initialSize - need to develop easy way of
        //differentiating these pre-allocated pages from "written-to" pages
        /*
        if (initalSize > 0) {
            // should expand to initialSize bytes
            size_t initialNumPages = (initalSize + pageSize_ -1) / pageSize_;
            int epoch = fm_->epoch();
            for (size_t pageNum = 0; pageNum < initialNumPages; ++pageNum) {
                Page page = addNewMultiPage(epoch);
                writeHeader(page,pageNum,epoch);
            }
        }
        */
    }

    FileBuffer::FileBuffer(FileMgr *fm, const size_t pageSize,const ChunkKey &chunkKey, const SQLTypeInfo sqlType, const size_t initialSize): AbstractBuffer(fm->getDeviceId(),sqlType), fm_(fm), metadataPages_(METADATA_PAGE_SIZE), chunkKey_(chunkKey)  {
        assert(fm_);
        calcHeaderBuffer();
        pageDataSize_ = pageSize_-reservedHeaderSize_;
    }




    FileBuffer::FileBuffer(FileMgr *fm,/* const size_t pageSize,*/ const ChunkKey &chunkKey, const std::vector<HeaderInfo>::const_iterator &headerStartIt, const std::vector<HeaderInfo>::const_iterator &headerEndIt): AbstractBuffer(fm->getDeviceId()), fm_(fm), metadataPages_(METADATA_PAGE_SIZE), /* pageSize_(pageSize),*/chunkKey_(chunkKey) {
        // We are being assigned an existing FileBuffer on disk

        assert(fm_);
        calcHeaderBuffer();
        //MultiPage multiPage(pageSize_); // why was this here?
        int lastPageId = -1;
        //Page lastMetadataPage;
        for (auto vecIt = headerStartIt; vecIt != headerEndIt; ++vecIt) {
            int curPageId = vecIt->pageId;

            // We only want to read last metadata page
            if (curPageId == -1) { //stats page
                metadataPages_.epochs.push_back(vecIt->versionEpoch);
                metadataPages_.pageVersions.push_back(vecIt->page);
            }
            else {
                if (curPageId != lastPageId) {
                    if (lastPageId == -1) {
                        // If we are on first real page
                        assert(metadataPages_.pageVersions.back().fileId != -1); // was initialized
                        readMetadata(metadataPages_.pageVersions.back());
                        pageDataSize_ = pageSize_-reservedHeaderSize_;

                    }
                    assert (curPageId == lastPageId + 1);
                    MultiPage multiPage(pageSize_);
                    multiPages_.push_back(multiPage);
                    lastPageId = curPageId;
                }
                multiPages_.back().epochs.push_back(vecIt->versionEpoch);
                multiPages_.back().pageVersions.push_back(vecIt->page);
            }
            if (curPageId == -1) { // meaning there was only a metadata page
                readMetadata(metadataPages_.pageVersions.back());
                pageDataSize_ = pageSize_-reservedHeaderSize_;
            }
        }
        //auto lastHeaderIt = std::prev(headerEndIt);
        //size_ = lastHeaderIt->chunkSize; 
    }

    FileBuffer::~FileBuffer() {
        // need to free pages
        // NOP
    }

    void FileBuffer::reserve(const size_t numBytes) {
        size_t numPagesRequested = (numBytes + pageSize_ -1) / pageSize_;
        size_t numCurrentPages = multiPages_.size();
        int epoch = fm_->epoch();

        for (size_t pageNum = numCurrentPages; pageNum < numPagesRequested; ++pageNum) {
            Page page = addNewMultiPage(epoch);
            writeHeader(page,pageNum,epoch);
        }
    }

    void FileBuffer::calcHeaderBuffer() {
        // 3 * sizeof(int) is for headerSize, for pageId and versionEpoch
        // sizeof(size_t) is for chunkSize
        //reservedHeaderSize_ = (chunkKey_.size() + 3) * sizeof(int) + sizeof(size_t);
        reservedHeaderSize_ = (chunkKey_.size() + 3) * sizeof(int);
        size_t headerMod = reservedHeaderSize_ % headerBufferOffset_;
        if (headerMod > 0) {
            reservedHeaderSize_ += headerBufferOffset_ - headerMod;
        }
        //pageDataSize_ = pageSize_-reservedHeaderSize_;
    }

    void FileBuffer::freePages() {
        // Need to zero headers (actually just first four bytes of header)
        
        // First delete metadata pages
        for (auto metaPageIt = metadataPages_.pageVersions.begin(); metaPageIt != metadataPages_.pageVersions.end(); ++metaPageIt) {
            FileInfo *fileInfo = fm_->getFileInfoForFileId(metaPageIt->fileId);
            fileInfo->freePage(metaPageIt->pageNum);
        }

        // Now delete regular pages
        for (auto multiPageIt = multiPages_.begin(); multiPageIt != multiPages_.end(); ++multiPageIt) {
            for (auto pageIt = multiPageIt->pageVersions.begin(); pageIt != multiPageIt->pageVersions.end(); ++pageIt) { 
                FileInfo *fileInfo = fm_->getFileInfoForFileId(pageIt->fileId);
                fileInfo->freePage(pageIt->pageNum);
            }
        }
    }


    void FileBuffer::read(int8_t * const dst, const size_t numBytes, const size_t offset, const MemoryLevel dstBufferType, const int deviceId ) {
        if (dstBufferType != CPU_LEVEL) {
            throw std::runtime_error("Unsupported Buffer type");
        }

        // variable declarations
        int8_t * curPtr = dst;    // a pointer to the current location in dst being written to
        size_t startPage = offset / pageDataSize_;
        size_t startPageOffset = offset % pageDataSize_;
        size_t numPagesToRead = (numBytes + startPageOffset + pageDataSize_ - 1) / pageDataSize_;
        assert (startPage + numPagesToRead <= multiPages_.size());
        size_t bytesLeft = numBytes;

        // Traverse the logical pages
        for (size_t pageNum = startPage; pageNum < startPage  + numPagesToRead; ++pageNum) {

            assert(multiPages_[pageNum].pageSize == pageSize_);
            Page page = multiPages_[pageNum].current();
            //printf("read: fileId=%d pageNum=%lu pageSize=%lu\n", page.fileId, page.pageNum, pageDataSize_);

            //FILE *f = fm_->files_[page.fileId]->f;
            FILE *f = fm_->getFileForFileId(page.fileId);
            assert(f);

            // Read the page into the destination (dst) buffer at its
            // current (cur) location
            size_t bytesRead;
            if (pageNum == startPage) {
                bytesRead = File_Namespace::read(f, page.pageNum * pageSize_ + startPageOffset + reservedHeaderSize_, min(pageDataSize_ - startPageOffset,bytesLeft), curPtr);
            }
            else {
                bytesRead = File_Namespace::read(f, page.pageNum * pageSize_ + reservedHeaderSize_, min(pageDataSize_,bytesLeft), curPtr);
            }
            curPtr += bytesRead;
            bytesLeft -= bytesRead;
        }
        assert (bytesLeft == 0);
    }

    void FileBuffer::copyPage(Page &srcPage, Page &destPage, const size_t numBytes, const size_t offset) { 
        //FILE *srcFile = fm_->files_[srcPage.fileId]->f;
        //FILE *destFile = fm_->files_[destPage.fileId]->f;
        assert(offset + numBytes < pageDataSize_);
        FILE *srcFile = fm_->getFileForFileId(srcPage.fileId); 
        FILE *destFile = fm_->getFileForFileId(destPage.fileId); 
        int8_t * buffer = new int8_t [numBytes]; 
        size_t bytesRead = File_Namespace::read(srcFile,srcPage.pageNum * pageSize_ + offset+reservedHeaderSize_, numBytes, buffer);
        assert(bytesRead == numBytes);
        size_t bytesWritten = File_Namespace::write(destFile,destPage.pageNum * pageSize_ + offset + reservedHeaderSize_, numBytes, buffer);
        assert(bytesWritten == numBytes);
        delete [] buffer;
    }

    Page FileBuffer::addNewMultiPage(const int epoch) {
        Page page = fm_->requestFreePage(pageSize_);
        MultiPage multiPage(pageSize_);
        multiPage.epochs.push_back(epoch);
        multiPage.pageVersions.push_back(page);
        multiPages_.push_back(multiPage);
        return page;
    }

    void FileBuffer::writeHeader(Page &page, const int pageId, const int epoch, const bool writeMetadata) {
        int intHeaderSize = chunkKey_.size() + 3; // does not include chunkSize
        vector <int> header (intHeaderSize);
        // in addition to chunkkey we need size of header, pageId, version
        //header[0] = (intHeaderSize - 1) * sizeof(int) + sizeof(size_t); // don't need to include size of headerSize value - sizeof(size_t) is for chunkSize
        header[0] = (intHeaderSize - 1) * sizeof(int); // don't need to include size of headerSize value - sizeof(size_t) is for chunkSize
        std::copy(chunkKey_.begin(), chunkKey_.end(), header.begin() + 1);
        header[intHeaderSize-2] = pageId;
        header[intHeaderSize-1] = epoch;
        FILE *f = fm_->getFileForFileId(page.fileId);
        size_t pageSize = writeMetadata ? METADATA_PAGE_SIZE : pageSize_;
        File_Namespace::write(f, page.pageNum*pageSize,(intHeaderSize) * sizeof(int),(int8_t *)&header[0]);
        /*
        if (writeSize) {
            File_Namespace::write(f, page.pageNum*pageSize_ + intHeaderSize*sizeof(int),sizeof(size_t),(int8_t *)&size_);
        }
        */
    }

    void FileBuffer::readMetadata(const Page &page) { 
        FILE *f = fm_->getFileForFileId(page.fileId); 
        fseek(f, page.pageNum*METADATA_PAGE_SIZE + reservedHeaderSize_, SEEK_SET);
        fread((int8_t *)&pageSize_,sizeof(size_t),1,f);
        fread((int8_t *)&size_,sizeof(size_t),1,f);
        vector <int> typeData (7); // assumes we will encode hasEncoder, bufferType, encodingType, encodingBits all as int
        fread((int8_t *)&(typeData[0]),sizeof(int),typeData.size(),f);
        hasEncoder = static_cast <bool> (typeData[0]);
        if (hasEncoder) {
            sqlType.set_type(static_cast<SQLTypes> (typeData[1]));
						sqlType.set_dimension(typeData[2]);
						sqlType.set_scale(typeData[3]);
						sqlType.set_notnull(static_cast<bool>(typeData[4]));
            sqlType.set_compression(static_cast<EncodingType> (typeData[5]));
            sqlType.set_comp_param(typeData[6]);
            initEncoder(sqlType);
            encoder->readMetadata(f);
        }
    }


    void FileBuffer::writeMetadata(const int epoch) {
        // Right now stats page is size_ (in bytes), bufferType, encodingType,
        // encodingDataType, numElements
        Page page = fm_->requestFreePage(METADATA_PAGE_SIZE);
        writeHeader(page,-1,epoch,true);
        FILE *f = fm_->getFileForFileId(page.fileId);
        fseek(f, page.pageNum*METADATA_PAGE_SIZE + reservedHeaderSize_, SEEK_SET);
        size_t numBytesWritten = fwrite((int8_t *)&pageSize_,sizeof(size_t),1,f);
        numBytesWritten = fwrite((int8_t *)&size_,sizeof(size_t),1,f);
        vector <int> typeData (7); // assumes we will encode hasEncoder, bufferType, encodingType, encodingBits all as int
        typeData[0] = static_cast<int>(hasEncoder); 
        if (hasEncoder) {
            typeData[1] = static_cast<int>(sqlType.get_type()); 
            typeData[2] = sqlType.get_dimension(); 
            typeData[3] = sqlType.get_scale(); 
            typeData[4] = static_cast<int>(sqlType.get_notnull()); 
            typeData[5] = static_cast<int>(sqlType.get_compression()); 
            typeData[6] = sqlType.get_comp_param(); 
        }
        numBytesWritten = fwrite((int8_t *)&(typeData[0]),sizeof(int),typeData.size(),f);
        if (hasEncoder) { // redundant
            encoder->writeMetadata(f);
         }
        metadataPages_.epochs.push_back(epoch);
        metadataPages_.pageVersions.push_back(page);
    }


    /*
    void FileBuffer::checkpoint() {
        if (isAppended_) {
            Page page = multiPages_[multiPages.size()-1].current();
            writeHeader(page,0,multiPages_[0].epochs.back());
        }
        isDirty_ = false;
        isUpdated_ = false;
        isAppended_ = false;
    }
    */

    void FileBuffer::append(int8_t * src, const size_t numBytes, const MemoryLevel srcBufferType, const int deviceId) {
        isDirty_ = true;
        isAppended_ = true;

        size_t startPage = size_ / pageDataSize_;
        size_t startPageOffset = size_ % pageDataSize_;
        size_t numPagesToWrite = (numBytes + startPageOffset + pageDataSize_ - 1) / pageDataSize_; 
        size_t bytesLeft = numBytes;
        int8_t * curPtr = src;    // a pointer to the current location in dst being written to
        size_t initialNumPages = multiPages_.size();
        size_ = size_ + numBytes;
        int epoch = fm_->epoch();
        for (size_t pageNum = startPage; pageNum < startPage  + numPagesToWrite; ++pageNum) {
            Page page;
            if (pageNum >= initialNumPages) {
                page = addNewMultiPage(epoch);
                writeHeader(page,pageNum,epoch);
            }
            else {
                // we already have a new page at current
                // epoch for this page - just grab this page
                page = multiPages_[pageNum].current();
            }
            assert(page.fileId >= 0); // make sure page was initialized
            FILE *f = fm_->getFileForFileId(page.fileId);
            size_t bytesWritten;
            if (pageNum == startPage) {
                bytesWritten = File_Namespace::write(f,page.pageNum*pageSize_ + startPageOffset + reservedHeaderSize_, min (pageDataSize_ - startPageOffset,bytesLeft),curPtr);
            }
            else {
                bytesWritten = File_Namespace::write(f, page.pageNum * pageSize_+reservedHeaderSize_, min(pageDataSize_,bytesLeft), curPtr);
            }
            curPtr += bytesWritten;
            bytesLeft -= bytesWritten;
            //if (pageNum == startPage + numPagesToWrite - 1) { // if last page
            //    //@todo make sure we stay consistent on append cases
            //    writeHeader(page,0,multiPages_[0].epochs.back());
            //}
        }
        assert (bytesLeft == 0);
    }

    void FileBuffer::write(int8_t * src,  const size_t numBytes, const size_t offset, const MemoryLevel srcBufferType, const int deviceId) {
        if (srcBufferType != CPU_LEVEL) {
            throw std::runtime_error("Unsupported Buffer type");
        }
        isDirty_ = true;
        if (offset < size_) {
            isUpdated_ = true;
        }
        bool tempIsAppended = false;

        if (offset + numBytes > size_) {
            tempIsAppended = true; // because isAppended_ could have already been true - to avoid rewriting header
            isAppended_ = true;
            size_ = offset + numBytes;
        }

        size_t startPage = offset / pageDataSize_;
        size_t startPageOffset = offset % pageDataSize_;
        size_t numPagesToWrite = (numBytes + startPageOffset + pageDataSize_ - 1) / pageDataSize_; 
        size_t bytesLeft = numBytes;
        int8_t * curPtr = src;    // a pointer to the current location in dst being written to
        size_t initialNumPages = multiPages_.size();
        int epoch = fm_->epoch();

        if (startPage > initialNumPages) { // means there is a gap we need to allocate pages for
            for (size_t pageNum = initialNumPages; pageNum < startPage; ++pageNum) {
                Page page = addNewMultiPage(epoch);
                writeHeader(page,pageNum,epoch);
            }
        }
        for (size_t pageNum = startPage; pageNum < startPage  + numPagesToWrite; ++pageNum) {
            Page page;
            if (pageNum >= initialNumPages) {
                page = addNewMultiPage(epoch);
                writeHeader(page,pageNum,epoch);
            }
            else if (multiPages_[pageNum].epochs.back() < epoch) { // need to create new page b/c this current one lags epoch and we can't overwrite it 
    // also need to copy if we are on first or last page
                Page lastPage = multiPages_[pageNum].current();
                page = fm_->requestFreePage(pageSize_);
                multiPages_[pageNum].epochs.push_back(epoch);
                multiPages_[pageNum].pageVersions.push_back(page);
                if (pageNum == startPage && startPageOffset > 0) {
                    // copyPage takes care of header offset so don't worry
                    // about it
                    copyPage(lastPage,page,startPageOffset,0);
                }
                if (pageNum == startPage + numPagesToWrite && bytesLeft > 0) { // bytesLeft should always > 0
                    copyPage(lastPage,page,pageDataSize_-bytesLeft,bytesLeft); // these would be empty if we're appending but we won't worry about it right now
                }
                writeHeader(page,pageNum,epoch);
            }
            else {
                // we already have a new page at current
                // epoch for this page - just grab this page
                page = multiPages_[pageNum].current();
            }
            assert(page.fileId >= 0); // make sure page was initialized
            FILE *f = fm_->getFileForFileId(page.fileId);
            size_t bytesWritten;
            if (pageNum == startPage) {
                bytesWritten = File_Namespace::write(f,page.pageNum*pageSize_ + startPageOffset + reservedHeaderSize_, min (pageDataSize_ - startPageOffset,bytesLeft),curPtr);
            }
            else {
                bytesWritten = File_Namespace::write(f, page.pageNum * pageSize_+reservedHeaderSize_, min(pageDataSize_,bytesLeft), curPtr);
            }
            curPtr += bytesWritten;
            bytesLeft -= bytesWritten;
            if (tempIsAppended && pageNum == startPage + numPagesToWrite - 1) { // if last page
                //@todo below can lead to undefined - we're overwriting num
                //bytes valid at checkpoint
                writeHeader(page,0,multiPages_[0].epochs.back(),true);
                //size_ = offset + numBytes;
            }
        }
        assert (bytesLeft == 0);
    }



} // File_Namespace

