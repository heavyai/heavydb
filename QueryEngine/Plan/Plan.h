/**
 * @file    Plan.h
 * @author  Steven Stewart <steve@map-d.com>
 */
#ifndef QueryEngine_Plan_Plan_h
#define QueryEngine_Plan_Plan_h

#include "../../DataMgr/Partitioner/Partitioner.h"
#include "../../DataMgr/Partitioner/TablePartitionMgr.h"
#include "../Parse/RA/ast/RelAlgNode.h"

using namespace Partitioner_Namespace;

namespace Plan_Namespace {

    class AbstractPlan {

    public:
        virtual int execute() = 0;
        virtual int optimize() = 0;
        virtual void* plan() = 0;
        virtual void print() = 0;
    
    };
    
    class QueryPlan : public AbstractPlan {
        
    public:
        QueryPlan(RA_Namespace::RelAlgNode *root);
        int execute();
        int optimize();
        void print();
        void* plan();
        
    private:
        RA_Namespace::RelAlgNode *root_;
        
    };
    
    class InsertPlan : public AbstractPlan {

    public:
        InsertPlan(const InsertData insertData_);
        ~InsertPlan();
        
        int execute();
        int optimize();
        void *plan();
        void print();
        
    private:
        InsertData data_;

    };

} // Plan_Namespace

#endif // QueryEngine_Plan_Plan_h