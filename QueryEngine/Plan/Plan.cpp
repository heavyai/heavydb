/**
 * @file    Plan.cpp
 * @author  Steven Stewart <steve@map-d.com>
 */
#include "Plan.h"

namespace Plan_Namespace {
    
    InsertPlan::InsertPlan(const InsertData &insertData_)
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
    
    void* InsertPlan::getPlan() {
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
    
    void* QueryPlan::getPlan() {
        return root_;
    }
    
    CreatePlan::CreatePlan(const std::string &tableName, const std::vector<std::string> &columnNames, const std::vector<mapd_data_t> columnTypes) : tableName_(tableName), columnNames_(columnNames), columnTypes_(columnTypes)
    {
        // NOP
    }
    
    int CreatePlan::execute() {
        return 0;
    }
    
    int CreatePlan::optimize() {
        return 0;
    }
    
    void* CreatePlan::getPlan() {
        return nullptr;
    }
    
    void CreatePlan::print() {
        
    }
    
} // Plan_Namespace