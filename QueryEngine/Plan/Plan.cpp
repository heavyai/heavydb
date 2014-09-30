/**
 * @file    Plan.cpp
 * @author  Steven Stewart <steve@map-d.com>
 */
#include "Plan.h"

namespace Plan_Namespace {
    
    InsertPlan::InsertPlan(const InsertData insertData_)
    : data_(insertData_)
    {
        // NOP
    }
    
    int InsertPlan::execute() {
        return 0;
    }
    
    int InsertPlan::optimize() {
        return 0;
    }
    
    void* InsertPlan::plan() {
        return &data_;
    }
    
    void InsertPlan::print() {
        // NOP
    }
    
    QueryPlan::QueryPlan(RA_Namespace::RelAlgNode *root) : root_(root) {
        // NOP
    }
    
    int QueryPlan::execute() {
        return 0;
    }
    
    int QueryPlan::optimize() {
        return 0;
    }
    
    void QueryPlan::print() {
        
    }
    
    void* QueryPlan::plan() {
        return root_;
    }
    
} // Plan_Namespace