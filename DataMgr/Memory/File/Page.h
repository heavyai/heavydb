/**
 * @file	Page.h
 * @author 	Steven Stewart <steve@map-d.com>
 * This file contains the declaration and definition of a Page type and a MultiPage type.
 */

#ifndef DATAMGR_MEMORY_FILE_PAGE_H
#define DATAMGR_MEMORY_FILE_PAGE_H

#include <cassert>
#include <deque>
#include <vector>
#include "../../../Shared/types.h"

namespace File_Namespace {
    
    /**
     * @struct Page
     * @brief A logical page (Page) belongs to a file on disk.
     *
     * A Page struct stores the file id for the file it belongs to, and it
     * stores its page number and number of used bytes within the page.
     *
     * Note: the number of used bytes should not be greater than the page
     * size. The page size is determined by the containing file.
     */
    struct Page {
        int fileId;				/// unique identifier of the owning file
        mapd_size_t pageNum;	/// page number
        
        /// Constructor
        Page(int fileId, mapd_size_t pageNum) : fileId(fileId), pageNum(pageNum) {}
        Page(): fileId(-1), pageNum(0) {}

        inline bool isValid() {return fileId >= 0;}
    };
    
    /**
     * @struct MultiPage
     * @brief The MultiPage stores versions of the same logical page in a deque.
     *
     * The purpose of MultiPage is to support storing multiple versions of the same
     * page, which may be located in different locations and in different files.
     * Associated with each version of a page is an "epoch" value, which is a temporal
     * reference.
     *
     *
     * Note that it should always be the case that version.size() == epoch.size().
     */
    struct MultiPage {
        mapd_size_t pageSize;
        std::deque<Page> pageVersions;
        std::deque<int> epochs;
        
        /// Constructor
        MultiPage(mapd_size_t pageSizeIn) :
		pageSize(pageSizeIn) {}
        
        /// Destructor -- purges all pages
        ~MultiPage() {
            while (pageVersions.size() > 0)
                pop();
        }
        
        /// Returns a reference to the most recent version of the page (optionally, the epoch
        /// is returned via the parameter "epoch").
        inline Page current(int *epoch = NULL) {
            if (pageVersions.size() < 1)
                throw std::runtime_error("No current version of the page exists in this MultiPage.");
            assert(pageVersions.size() > 0); // @todo should use proper exception handling
            if (epoch != NULL)
                *epoch = this->epochs.back();
            return pageVersions.back();
        }
        
        /// Pushes a new page with epoch value
        inline void push(Page &page, const int epoch) {
            pageVersions.push_back(page);
            epochs.push_back(epoch);
            assert(pageVersions.size() == epochs.size());
        }
        
        /// Purges the oldest Page
        inline void pop() {
            if (pageVersions.size() < 1)
                throw std::runtime_error("No page to pop.");
            pageVersions.pop_front();
            epochs.pop_front();
            assert(pageVersions.size() == epochs.size());
        }
    };

    /**
     * @type HeaderInfo 
     * @brief Stores Pair of ChunkKey and Page id and version, in a pair with
     * a Page struct itself (File id and Page num)
     */

    typedef std::pair<std::pair<std::vector<int>,std::vector<int> >, Page > HeaderInfo;
    
} // File_Namespace

#endif // DATAMGR_MEMORY_FILE_PAGE_H
