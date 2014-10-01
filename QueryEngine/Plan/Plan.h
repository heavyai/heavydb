/**
 * @file    Plan.h
 * @author  Steven Stewart <steve@map-d.com>
 */
#ifndef QueryEngine_Plan_Plan_h
#define QueryEngine_Plan_Plan_h

#include "../../DataMgr/Partitioner/Partitioner.h"
#include "../../DataMgr/Partitioner/TablePartitionMgr.h"
#include "../Parse/RA/ast/RelAlgNode.h"
#include "../Parse/SQL/ast/MapdDataT.h"

using namespace Partitioner_Namespace;

namespace Plan_Namespace {

    /**
     * This class specifies an interface for a "Plan" object. A plan
     * represents a "program" or specification for carrying out some
     * task; for example, a scan (QueryPlan), an insert (InsertPlan),
     * etc. (This is a pure virtual class -- an abstract interface.)
     *
     * The rationale behind these abstractions is that it permits
     * query plans of different forms: for example, a tree or other
     * kind of data structure.
     */
    class AbstractPlan {

    public:
        
        /**
         * @brief Executes the plan.
         * @return Non-zero for success; otherwise, an error code
         */
        virtual int execute() = 0;
        
        /**
         * @brief Optimizes the plan.
         * @return Non-zero for success; otherwise, an error code
         */
        virtual int optimize() = 0;
        
        /**
         * @brief Returns a void pointer to the underlying plan.
         */
        virtual void* getPlan() = 0;
        
        /**
         * Prints a representation of the plan to stdout.
         */
        virtual void print() = 0;
    };
    
    /**
     * A "query plan" encodes the instructions needed to execute a
     * database scan in the form of an operator tree (mainly,
     * conventional relational algebra operators). A post-order
     * traversal of the tree (i.e., visit the operands before the
     * operator) is suitable for executing the plan. The plan can
     * also be passed to an optimizer in order to produce a lower
     * cost plan (if possible). QueryPlan implements the AbstractPlan
     * interface.
     */
    class QueryPlan : public AbstractPlan {
        
    public:
        QueryPlan(RA_Namespace::RelAlgNode *root);
        int execute();
        int optimize();
        void print();
        void* getPlan();
        
    private:
        RA_Namespace::RelAlgNode *root_;
        
    };
    
    /**
     * An "insert plan" encodes the information necessary in order
     * to insert a tuple into a relational table. It implements the
     * the AbstractPlan interface.
     */
    class InsertPlan : public AbstractPlan {

    public:
        InsertPlan(const InsertData &insertData_);
        ~InsertPlan();
        
        int execute();
        int optimize();
        void *getPlan();
        void print();
        
    private:
        InsertData data_;

    };
    
    /**
     * A "create plan" encodes the information necessary in order
     * to create table in the relational database.
     */
    class CreatePlan : public AbstractPlan {

    public:
        CreatePlan(const std::string &tableName, const std::vector<std::string> &columnNames, const std::vector<mapd_data_t> columnTypes);
        ~CreatePlan();
        
        int execute();
        int optimize();
        void *getPlan();
        void print();
        
    private:
        std::string tableName_;
        std::vector<std::string> columnNames_;
        std::vector<mapd_data_t> columnTypes_;
    };

    /**
     * A "drop plan" encodes the information necessary in order
     * to drop a table from the relational database.
     */
    class DropPlan : public AbstractPlan {
        
    public:
        DropPlan(const std::string &tableName);
        ~DropPlan();
        
        int execute();
        int optimize();
        void *getPlan();
        void print();
        
    private:
        std::string tableName_;
    };
    
    /**
     * A "delete plan" encodes the information necessary in order
     * to delete tuples from a table in the relational database.
     */
    class DeletePlan : public AbstractPlan {
        
    public:
        DeletePlan(const std::string &tableName);
        DeletePlan();
        
        int execute();
        int optimize();
        void *getPlan();
        void print();
        
    private:
        std::string tableName_;
    };
    
} // Plan_Namespace

#endif // QueryEngine_Plan_Plan_h