/**
 * @file	ScanOp.h
 * @author	Steven Stewart <steve@map-d.com>
 */
#ifndef QUERYENGINE_PARSE_RA_SCANOP_H
#define QUERYENGINE_PARSE_RA_SCANOP_H

#include <cassert>
#include "UnaryOp.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {
    
    class ScanOp : public UnaryOp {
        
    public:
        RelExpr *n1 = nullptr;
        AttrList *n2 = nullptr;
        Predicate *n3 = nullptr;
        
        /// Constructor
        ScanOp(RelExpr *n1, AttrList *n2, Predicate *n3) {
            assert(n1 && n2 && n3);
            this->n1 = n1;
            this->n2 = n2;
            this->n3 = n3;
        }
        
        virtual void accept(class Visitor &v) {
            v.visit(this);
        }
        
    };
    
} // RA_Namespace

#endif // QUERYENGINE_PARSE_RA_SCANOP_H
