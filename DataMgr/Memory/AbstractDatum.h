/**
 * @file    AbstractDatum.h
 * @author  Steven Stewart <steve@map-d.com>
 * @author  Todd Mostak <todd@map-d.com>
 */
#ifndef DATAMGR_MEMORY_ABSTRACTDATUM_H
#define DATAMGR_MEMORY_ABSTRACTDATUM_H

#include "../../Shared/types.h"

namespace Memory_Namespace {
    
    /**
     * @class   AbstractDatum
     * @brief   An AbstractDatum is a unit of data management for a data manager.
     */
    class AbstractDatum {
        
    public:
        virtual ~AbstractDatum() {}
        
        virtual void read(mapd_addr_t const dst, const mapd_size_t offset, const mapd_size_t nbytes = 0) = 0;
        virtual void write(mapd_addr_t src, const mapd_size_t offset, const mapd_size_t nbytes) = 0;
        virtual void append(mapd_addr_t src, const mapd_size_t nbytes) = 0;
        
        virtual mapd_size_t pageCount() const = 0;
        virtual mapd_size_t size() const = 0;
        virtual mapd_size_t used() const = 0;
        virtual bool isDirty() const = 0;

    };
    
} // Memory_Namespace

#endif // DATAMGR_MEMORY_ABSTRACTDATUM_H
