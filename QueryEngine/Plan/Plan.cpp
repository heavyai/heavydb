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
        
        throw std::runtime_error( "execute() for InsertPlan is currently unsupported" );
    }
    
    int InsertPlan::optimize() {
        throw std::runtime_error( "optimize() for InsertPlan is currently unsupported" );
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
        throw std::runtime_error( "execute() for QueryPlan is currently unsupported" );
    }
    
    int QueryPlan::optimize() {
        throw std::runtime_error( "optimize() for QueryPlan is currently unsupported" );
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
        throw std::runtime_error( "execute() for CreatePlan is currently unsupported" );
    }
    
    int CreatePlan::optimize() {
        throw std::runtime_error( "optimize() for CreatePlan is currently unsupported" );
    }
    
    void* CreatePlan::getPlan() {
        return nullptr;
    }
    
    void CreatePlan::print() {
        
    }
    
    DropPlan::DropPlan(const std::string &tableName) : tableName_(tableName) {
        // NOP
    }
    
    int DropPlan::execute() {
        throw std::runtime_error( "execute() for DropPlan is currently unsupported" );
    }
    
    int DropPlan::optimize() {
        throw std::runtime_error( "optimize() for DropPlan is currently unsupported" );
    }
    
    void* DropPlan::getPlan() {
        return nullptr;
    }
    
    void DropPlan::print() {
        
    }
    
    DeletePlan::DeletePlan(const std::string &tableName) : tableName_(tableName)
    {
        this->print();
    }
    
    int DeletePlan::execute() {
        throw std::runtime_error( "execute() for DeletePlan is currently unsupported" );
    }
    
    int DeletePlan::optimize() {
        throw std::runtime_error( "optimize() for DeletePlan is currently unsupported" );
    }
    
    void* DeletePlan::getPlan() {
        return nullptr;
    }
    
    void DeletePlan::print() {
        std::cout << "DELETE FROM " << tableName_.c_str() << ";" << std::endl;
    }
    
    AlterPlan::AlterPlan(const std::string &tableName, const std::string &columnName, const mapd_data_t columnType, bool drop) : tableName_(tableName), columnName_(columnName), columnType_(columnType), drop_(drop)
    {
        // NOP
    }

    
    int AlterPlan::execute() {
        throw std::runtime_error( "execute() for AlterPlan is currently unsupported" );
    }
    
    int AlterPlan::optimize() {
        throw std::runtime_error( "optimize() for AlterPlan is currently unsupported" );
    }
    
    void* AlterPlan::getPlan() {
        return nullptr;
    }
    
    void AlterPlan::print() {
        
    }
    
    RenamePlan::RenamePlan(const std::string &oldTableName, const std::string &newTableName)
    : oldTableName_(oldTableName), newTableName_(newTableName)
    {
        // NOP
    }
    
    int RenamePlan::execute() {
        throw std::runtime_error( "execute() for RenamePlan is currently unsupported" );
    }
    
    int RenamePlan::optimize() {
        throw std::runtime_error( "optimize() for RenamePlan is currently unsupported" );
    }
    
    void* RenamePlan::getPlan() {
        throw std::runtime_error( "getPlan() for RenamePlan is currently unsupported" );
    }
    
    void RenamePlan::print() {
        throw std::runtime_error( "print() for RenamePlan is currently unsupported" );
    }

} // Plan_Namespace