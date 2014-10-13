/**
 * @file    Translator.h
 * @author  Steven Stewart <steve@map-d.com>
 */
#include "Translator.h"

// SQL nodes
#include "../Parse/SQL/ast/AggrExpr.h"
#include "../Parse/SQL/ast/AlterStmt.h"
#include "../Parse/SQL/ast/Column.h"
#include "../Parse/SQL/ast/ColumnDef.h"
#include "../Parse/SQL/ast/ColumnDefList.h"
#include "../Parse/SQL/ast/ColumnList.h"
#include "../Parse/SQL/ast/Comparison.h"
#include "../Parse/SQL/ast/CreateStmt.h"
#include "../Parse/SQL/ast/DdlStmt.h"
#include "../Parse/SQL/ast/DeleteStmt.h"
#include "../Parse/SQL/ast/DmlStmt.h"
#include "../Parse/SQL/ast/DropStmt.h"
#include "../Parse/SQL/ast/FromClause.h"
#include "../Parse/SQL/ast/InsertColumnList.h"
#include "../Parse/SQL/ast/InsertStmt.h"
#include "../Parse/SQL/ast/Literal.h"
#include "../Parse/SQL/ast/LiteralList.h"
#include "../Parse/SQL/ast/MapdDataT.h"
#include "../Parse/SQL/ast/MathExpr.h"
#include "../Parse/SQL/ast/OptAllDistinct.h"
#include "../Parse/SQL/ast/OptGroupby.h"
#include "../Parse/SQL/ast/OptHaving.h"
#include "../Parse/SQL/ast/OptOrderby.h"
#include "../Parse/SQL/ast/OptLimit.h"
#include "../Parse/SQL/ast/OptWhere.h"
#include "../Parse/SQL/ast/OrderbyColumn.h"
#include "../Parse/SQL/ast/OrderByColumnList.h"
#include "../Parse/SQL/ast/Predicate.h"
#include "../Parse/SQL/ast/RenameStmt.h"
#include "../Parse/SQL/ast/ScalarExpr.h"
#include "../Parse/SQL/ast/ScalarExprList.h"
#include "../Parse/SQL/ast/SearchCondition.h"
#include "../Parse/SQL/ast/Selection.h"
#include "../Parse/SQL/ast/SelectStmt.h"
#include "../Parse/SQL/ast/SqlStmt.h"
#include "../Parse/SQL/ast/Table.h"
#include "../Parse/SQL/ast/TableList.h"

// RA nodes
#include "../Parse/RA/ast/AggrExpr.h"
#include "../Parse/RA/ast/AggrList.h"
#include "../Parse/RA/ast/AntijoinOp.h"
#include "../Parse/RA/ast/Attribute.h"
#include "../Parse/RA/ast/AttrList.h"
#include "../Parse/RA/ast/Comparison.h"
#include "../Parse/RA/ast/DiffOp.h"
#include "../Parse/RA/ast/Expr.h"
#include "../Parse/RA/ast/ExtendOp.h"
#include "../Parse/RA/ast/GroupbyOp.h"
#include "../Parse/RA/ast/JoinOp.h"
#include "../Parse/RA/ast/MathExpr.h"
#include "../Parse/RA/ast/OuterjoinOp.h"
#include "../Parse/RA/ast/Predicate.h"
#include "../Parse/RA/ast/ProductOp.h"
#include "../Parse/RA/ast/Program.h"
#include "../Parse/RA/ast/ProjectOp.h"
#include "../Parse/RA/ast/Relation.h"
#include "../Parse/RA/ast/RelExpr.h"
#include "../Parse/RA/ast/RelExprList.h"
#include "../Parse/RA/ast/RenameOp.h"
#include "../Parse/RA/ast/ScanOp.h"
#include "../Parse/RA/ast/SelectOp.h"
#include "../Parse/RA/ast/SemijoinOp.h"
#include "../Parse/RA/ast/SortOp.h"
#include "../Parse/RA/ast/UnionOp.h"

using namespace RA_Namespace;

namespace Plan_Namespace {

    Translator::Translator(Catalog &c) : c_(c) {
        insertData_.numRows = 0;
        insertData_.tableId = -1;
    }
    
    // it's important to update this function to clear the state when adding
    // support for additional SQL statements/features
    void Translator::clearState() {
        queryTables_.clear();
        queryColumns_.clear();
        queryPredicate_ = nullptr;
        
        insertData_.numRows = 0;
        insertData_.tableId = -1;
        insertTable_ = nullptr;
        insertColumns_.clear();
        insertValues_.clear();
        byteCount_ = 0;
        
        deleteTableName_ = nullptr;
        deletePredicate_ = nullptr;
        
        updateTableName_ = nullptr;
        updateColumns_.clear();
        updateValues_.clear();
        
        createTableName_ = nullptr;
        createColumns_.clear();
        createTypes_.clear();
        
        tableNames_.clear();
        columnNames_.clear();
        
        error_ = false;
        errorMsg_ = "";
    }
    
    AbstractPlan* Translator::translate(SQL_Namespace::ASTNode *parseTreeRoot) {
        assert(parseTreeRoot);

        // clear private data structures
        clearState();
        
        // translate (visit) the parse tree
        AbstractPlan *queryPlan = nullptr;
        parseTreeRoot->accept(*this);
        
        /*
         for (size_t i = 0; i < tableNames_.size(); ++i)
            printf("tableNames_[%zu] = %s\n", i, tableNames_[i].c_str());
         for (size_t i = 0; i < columnNames_.size(); ++i)
            printf("columnNames[%zu] = %s\n", i, columnNames_[i].second.c_str());
         */
        
        // translate the SQL AST to an RA query plan tree
        if (stmtType_ == QUERY_STMT) {
            annotateQuery();
            if (error_)
                return nullptr;
            // @todo type check expressions
            queryPlan = translateQuery();
        }
        else if (stmtType_ == INSERT_STMT) {
            translateInsert();
            if (error_)
                return nullptr;
            assert(insertData_.numRows > 0);
            return new InsertPlan(insertData_); // "plan" is accessed via call to getInsertData()
        }
        else if (stmtType_ == CREATE_STMT) {
            queryPlan = translateCreate();
        }
        else if (stmtType_ == DROP_STMT) {
            queryPlan = translateDrop();
        }
        else if (stmtType_ == DELETE_STMT) {
            queryPlan = translateDelete();
        }
        else if (stmtType_ == ALTER_STMT) {
            queryPlan = translateAlter();
        }
        else if (stmtType_ == RENAME_STMT)
            queryPlan = translateRename();
        else
            throw std::runtime_error("Unable to translate SQL statement to RA query plan");

        // check for error state
        if (error_)
            return nullptr;
        
        return queryPlan; // returns a Project-Select query tree (Scan)
    }
    
    RA_Namespace::Predicate* Translator::translatePredicate(const SQL_Namespace::Predicate &sqlPred) {

        /*if (sqlPred.n3) {
            sqlPred.n3->accept(*this); // Comparison
        }
        else {
            sqlPred.n1->accept(*this); // Predicate
            sqlPred.n2->accept(*this); // Predicate
        }*/

        return nullptr;
    }

    /**
     * The translate query method translates an SQL query (i.e., select statement)
     * into a query plan, a pointer to which is returned. It does so by a multi-step
     * procedure, outlined in body of the method. The procedure was adapted from
     * Sciore's Database Design and Implementation text book.
     */
    QueryPlan* Translator::translateQuery() {
        assert(queryTables_.size() > 0);

        // Step 1:  create Relation nodes for each table
        std::vector<Relation*> relations;
        size_t numTables = queryTables_.size();
        for (size_t i = 0; i < numTables; ++i)
            relations.push_back(new Relation(queryTables_[i]->metadata));
        
        // Step 2:  take the product of the relations from Step 1
        ProductOp* productOfRelations = nullptr;
        if (numTables > 1) {
            productOfRelations = new ProductOp((RelExpr*)relations[0], (RelExpr*)relations[1]);
            for (size_t i = 2; i < numTables; ++i)
                productOfRelations = new ProductOp((RelExpr*)productOfRelations, (RelExpr*)relations[i]);
        }
        
        // Step 3:  select on the predicate in the where clause
        SelectOp *select = nullptr;
        if (numTables == 1 && queryPredicate_) {
            select = new SelectOp((RelExpr*)relations[0], translatePredicate(queryPredicate_));
        }
        else if (queryPredicate_) {
            assert(numTables > 1);
            select = new SelectOp((RelExpr*)productOfRelations, translatePredicate(queryPredicate_));
        }
        
        // Step 4: project on the fields in the selection clause
        size_t numFields = queryColumns_.size();
        assert(numFields > 0);

        AttrList *fields = new AttrList(new Attribute(queryColumns_[0]->metadata));
        for (size_t i = 1; i < numFields; ++i) {
            assert(queryColumns_[0]->name.second != "");
            fields = new AttrList(fields, new Attribute(queryColumns_[i]->metadata));
        }
        
        ProjectOp *project = nullptr;
        if (select) {
            project = new ProjectOp((RelExpr*)select, fields);
        }
        else {
            if (numTables == 1)
                project = new ProjectOp((RelExpr*)relations[0], fields);
            else
                project = new ProjectOp((RelExpr*)productOfRelations, fields);
        }
        
        // Step 6:  return
        assert(project);
        return new QueryPlan(project);
    }
    
    /**
     * This method annonates the AST for a query (i.e., select statement). It traverses
     * the list of nodes collected in the queryTables_ vector, requesting metadata for
     * table from the system catalog (@see Catalog). It also obtains the necessary column
     * metadata for the gathered columns. The Translator's error state is updated if
     * a table or column is not found, or if a column name is ambiguous due to appearing
     * in more than one the tables in queryTables_.
     */
    void Translator::annotateQuery() {
        // retieve table metadata from Catalog
        // set error if a table does not exist
        TableRow tableMetadata;
        for (size_t i = 0; i < queryTables_.size(); ++i) {
            mapd_err_t err = c_.getMetadataForTable(queryTables_[i]->name.second, tableMetadata);
            if (err != MAPD_SUCCESS) {
                error_ = true;
                errorMsg_ = "Table '" + tableNames_[i] + "' not found";
                return;
            }
            queryTables_[i]->metadata = tableMetadata;
        }
        
        // retrieve column metadata from Catalog
        // set error if column does not exist or there is ambiguity
        std::vector<ColumnRow> columnMetadata;
        mapd_err_t err = c_.getMetadataForColumns(tableNames_, columnNames_, columnMetadata);
        if (err != MAPD_SUCCESS) {
            error_ = true;
            errorMsg_ = "Catalog error";
            if (err == MAPD_ERR_COLUMN_DOES_NOT_EXIST) {
                std::string col = columnNames_[columnMetadata.size()].second;
                errorMsg_ = "Column '" + col + "' not found";
            }
            else if (err == MAPD_ERR_COLUMN_IS_AMBIGUOUS) {
                std::string col = columnNames_[columnMetadata.size()].second;
                errorMsg_ = "Ambiguous column '" + col + "'";
            }
            return;
        }
        assert(columnMetadata.size() == columnNames_.size());
        
        // annotate SQL column nodes with Catalog metadata
        for (size_t i = 0; i < queryColumns_.size(); ++i)
            queryColumns_[i]->metadata = columnMetadata[i];
    }
    
    InsertPlan* Translator::translateInsert() {
        assert(insertValues_.size() == insertColumns_.size());
        mapd_err_t err = MAPD_SUCCESS;
        
        // presently, only 1 row inserted at a time
        // @todo Support for bulk insert instead of just one row at a time
        insertData_.numRows = 1;
        
        // set table id for insertData_ (obtain table metadata from Catalog)
        err = c_.getMetadataForTable(insertTable_->name.second, insertTable_->metadata);
        if (err != MAPD_SUCCESS) {
            error_ = true;
            errorMsg_ = "Table '" + insertTable_->name.second + "' not found";
            return nullptr;
        }
        insertData_.tableId = insertTable_->metadata.tableId;
        
        // set column ids for insertData_ (obtain column metadata from Catalog)
        std::vector<std::string> insertColumnNames;
        for (size_t i  = 0; i < insertColumns_.size(); ++i)
            insertColumnNames.push_back(insertColumns_[i]->name);
        assert(insertColumns_.size() == insertColumnNames.size());
        
        std::vector<ColumnRow> insertColumnMetadata;
        err = c_.getMetadataForColumns(insertTable_->name.second, insertColumnNames, insertColumnMetadata);
        assert(insertColumns_.size() == insertColumnMetadata.size());
        
        for (size_t i  = 0; i < insertColumnMetadata.size(); ++i) {
            insertData_.columnIds.push_back(insertColumnMetadata[i].columnId);
            insertColumns_[i]->metadata = insertColumnMetadata[i];
        }

        // type check insert statement
        typeCheckInsert();
        if (isError())
            return nullptr;
        
        // package the data to be inserted
        for (size_t i = 0; i < insertValues_.size(); ++i) {
            mapd_byte_t *data = new mapd_byte_t[byteCount_];
            mapd_byte_t *pData = data;
            mapd_size_t currentByte = 0;
            
            // printf("[%zu] int=%d float=%f\n", i, insertValues_[i]->intData, insertValues_[i]->realData);
            
            // for each possible type, the bytes are copied one at a time into
            // the bytes of the mapd_byte_t data pointer
            if (insertValues_[i]->type == INT_TYPE) {
                mapd_byte_t *tmp = (mapd_byte_t*)&insertValues_[i]->intData;
                for (int j = 0; j < sizeof(int); ++j, ++tmp, ++pData)
                    *pData = *tmp;
                // printf("%d == %d\n", *((int*)(data+currentByte)), literalNodes_[i]->intData);
                assert(*((int*)(data+currentByte)) == insertValues_[i]->intData);
                currentByte += sizeof(int);
            }
            else if (insertValues_[i]->type == FLOAT_TYPE) {
                mapd_byte_t *tmp = (mapd_byte_t*)&insertValues_[i]->realData;
                for (int j = 0; j < sizeof(float); ++j, ++tmp, ++pData)
                    *pData = *tmp;
                // printf("%f == %f\n", *((float*)(data+currentByte)), literalNodes_[i]->realData);
                assert(*((float*)(data+currentByte)) == insertValues_[i]->realData);
                currentByte += sizeof(int);
            }
            
            // push the packaged data into insertData_
            insertData_.data.push_back((void*)data);
        }
        
        return new InsertPlan(insertData_);
    }
    
    void Translator::typeCheckInsert() {
        for (size_t i = 0; i < insertColumns_.size(); ++i) {
            if (insertColumns_[i]->metadata.columnType == INT_TYPE && insertValues_[i]->type == FLOAT_TYPE) {
                error_ = true;
                errorMsg_ = "Type mismatch at column '" + insertColumns_[i]->name +"' (cannot downcast float to int)";
                return;
            }
            else if (insertColumns_[i]->metadata.columnType == FLOAT_TYPE && insertValues_[i]->type == INT_TYPE) {
                insertValues_[i]->realData = (float)insertValues_[i]->intData;
                insertValues_[i]->type = FLOAT_TYPE;
            }
        }
    }
    
    CreatePlan* Translator::translateCreate() {
        assert(createTableName_);
        assert(createColumns_.size() > 0);
        assert(createColumns_.size() == createTypes_.size());
        
        // set table name
        std::string tableName = createTableName_->name.first;
    
        // insert column names
        std::vector<std::string> columnNames;
        for (int i = 0; i < columnNames_.size(); ++i)
            columnNames.push_back(columnNames_[i].second);
        
        // insert column types
        std::vector<mapd_data_t> columnTypes;
        for (int i = 0; i < createTypes_.size(); ++i)
            columnTypes.push_back(createTypes_[i]->type);
        
        // return the plan
        return new CreatePlan(tableName, columnNames, columnTypes);
    }
    
    DropPlan* Translator::translateDrop() {
        // printf("[%s] [%s]\n", dropTableName_->name.first.c_str(), dropTableName_->name.second.c_str());
        return new DropPlan(dropTableName_->name.second);
    }
    
    DeletePlan* Translator::translateDelete() {
        return new DeletePlan(deleteTableName_->name.second);
    }
    
    AlterPlan* Translator::translateAlter() {
        assert(tableNames_.size() == 1 && columnNames_.size() == 1);
        return new AlterPlan(tableNames_[0], columnNames_[0].second, alterColumnType_, alterDrop_);
    }

    RenamePlan* Translator::translateRename() {
        assert(tableNames_.size() == 1);
        return new RenamePlan(tableNames_[0], renameTableNewName_);
    }
    
    RA_Namespace::Predicate* Translator::translatePredicate(SQL_Namespace::Predicate* v) {
        assert(v);
        
        if (v->op != "" && v->n1 && !v->n2) { // unary op with predicate
            return new RA_Namespace::Predicate(v->op, (RA_Namespace::Predicate*)translatePredicate(v->n1));
        }
        else if (v->op != "" && v->n1 && v->n2) { // binary op with predicate
            return new RA_Namespace::Predicate(v->op, (RA_Namespace::Predicate*)translatePredicate(v->n1), (RA_Namespace::Predicate*)translatePredicate(v->n2));
        }
        else if (v->op == "" && v->n1 && !v->n2) { // predicate nested in brackets
            return new RA_Namespace::Predicate((RA_Namespace::Predicate*)translatePredicate(v->n1));
        }
        else if (v->n3) { // Comparison
            return new RA_Namespace::Predicate(translateComparison(v->n3));
        }
        else
            throw std::runtime_error("Unsupported SQL statement.");
    }
    
    RA_Namespace::Comparison* Translator::translateComparison(SQL_Namespace::Comparison* v) {
        assert(v);
        
        OpType compOp; // @todo OpType should be available to both SQL and RA, and these checks below would then be unnecessary. Definitely should fix this, but for now it should work.
        
        if (v->op == ">")
            compOp = OP_GT;
        else if (v->op == "<")
            compOp = OP_LT;
        else if (v->op == ">=")
            compOp = OP_GTE;
        else if (v->op == "<=")
            compOp = OP_LTE;
        else if (v->op == "!=")
            compOp = OP_NEQ;
        else if (v->op == "=")
            compOp = OP_EQ;
        else
            throw std::runtime_error("Unsupported operator in comparison.");
        
        if (v->n1 &&  v->n2)
            return new RA_Namespace::Comparison(compOp, translateMathExpr(v->n1), translateMathExpr((v->n2)));
        else
            throw std::runtime_error("Unsupported SQL statement.");
    }
    
    RA_Namespace::MathExpr* Translator::translateMathExpr(SQL_Namespace::MathExpr* v) {
        assert(v);
        
        // numeric type
        if (v->numericFlag) {
            if (v->intFlag)
                return new RA_Namespace::MathExpr(v->intVal);
            else if (v->floatFlag)
                return new RA_Namespace::MathExpr(v->floatVal);
            else
                throw std::runtime_error("Unsupported data type in math expression.");
        }
        
        // Column
        else if (v->n3) {
            return new RA_Namespace::MathExpr(translateColumn(v->n3));
        }
        
        // ( MathExpr )
        else if (v->n0)
            return new RA_Namespace::MathExpr(translateMathExpr(v->n0));
        
        // MathExpr op MathExpr
        else if (v->op != "" && v->n1 && v->n2) {
            OpType mathOp;
            if (v->op == "/")
                mathOp = OP_DIVIDE;
            else if (v->op == "*")
                mathOp = OP_MULTIPLY;
            else if (v->op == "+")
                mathOp = OP_ADD;
            else if (v->op == "-")
                mathOp = OP_SUBTRACT;
            else
                throw std::runtime_error("Unsupported operator in math expression.");
            
            return new RA_Namespace::MathExpr(mathOp, translateMathExpr(v->n1), translateMathExpr(v->n2));
        }
        else
            throw std::runtime_error("Unsupported SQL statement.");
    }
    
    RA_Namespace::Attribute* Translator::translateColumn(SQL_Namespace::Column* v) {
        assert(v);
        
        // obtain column metadata from Catalog
        std::vector<std::pair<std::string, std::string>> columnNames;
        std::vector<ColumnRow> columnRows;
        columnNames.push_back(v->name);
        c_.getMetadataForColumns(tableNames_, columnNames, columnRows);
        
        assert(columnRows.size() == 1);
        return new RA_Namespace::Attribute(columnRows[0]);
    }
    
    void* Translator::translateMapdDataT(SQL_Namespace::MapdDataT* v) {
        assert(v);
        return nullptr;
    }
    
    mapd_data_t typeCheckMathExpr(SQL_Namespace::MathExpr *v) {
        assert(v);
        
        if (v->numericFlag) { // base case
            if (v->intFlag)
                return INT_TYPE;
            else if (v->floatFlag)
                return FLOAT_TYPE;
            else
                throw std::runtime_error("Unsupported data type.");
        }
        else if (v->n0)
            return typeCheckMathExpr(v->n0);
        else if (v->n1 && v->n2) {
            mapd_data_t left = typeCheckMathExpr(v->n1);
            mapd_data_t right = typeCheckMathExpr(v->n2);
            assert(left == INT_TYPE || left == FLOAT_TYPE);
            assert(right == INT_TYPE || right == FLOAT_TYPE);
            
            if (left == FLOAT_TYPE || right == FLOAT_TYPE)
                return FLOAT_TYPE;
            else
                return INT_TYPE;
        }
        
        throw std::runtime_error("Type-checking error.");
    }
    
    void Translator::visit(AlterStmt *v) {
        stmtType_ = ALTER_STMT;
        if (v->n1) v->n1->accept(*this); // Table
        if (v->n2) v->n2->accept(*this); // Column
        if (v->n3) v->n3->accept(*this); // MapdDataT
    }
    
    void Translator::visit(Column *v) {
        columnNames_.push_back(v->name);
        
        if (stmtType_ == QUERY_STMT) {
            if (queryInsidePredicate_)
                queryPredicateColumnNames_.push_back(v->name);
            else
                queryColumns_.push_back(v);
        }
        else if (stmtType_ == CREATE_STMT)
            createColumns_.push_back(v);
        else if (stmtType_ == ALTER_STMT)
            ; // NOP; collected columnNames_ vector is sufficient
        else
            throw std::runtime_error("Unsupported SQL feature.");
    }
    
    void Translator::visit(ColumnDef *v) {
        // printf("<ColumnDef>\n");
        if (v->n1) v->n1->accept(*this); // Column
        if (v->n2) v->n2->accept(*this); // MapdDataT
    }
    
    void Translator::visit(ColumnDefList *v) {
        // printf("<ColumnDefList>\n");
        if (v->n1) v->n1->accept(*this); // ColumnDefList
        if (v->n2) v->n2->accept(*this); // ColumnDef
    }
    
    void Translator::visit(ColumnList *v) {
        // printf("<ColumnList>\n");
        if (v->n1) v->n1->accept(*this); // ColumnList
        if (v->n2) v->n2->accept(*this); // Column
    }
    
    void Translator::visit(SQL_Namespace::Comparison *v) {
        // printf("<Comparison>\n");
        if (v->op != "" && v->n1 && v->n2) {
            v->n1->accept(*this); // MathExpr
            v->n2->accept(*this); // MathExpr
        }
        else
            throw std::runtime_error("Unsupported SQL feature.");
    }
    
    void Translator::visit(CreateStmt *v) {
        // printf("<CreateStmt>\n");
        stmtType_ = CREATE_STMT;
        if (v->n1) v->n1->accept(*this); // Table
        if (v->n2) v->n2->accept(*this); // ColumnDefList
    }

    void Translator::visit(DdlStmt *v) {
        // printf("<DdlStmt>\n");
        if (v->n1) v->n1->accept(*this); // CreateStmt
        if (v->n2) v->n2->accept(*this); // DropStmt
        if (v->n3) v->n3->accept(*this); // AlterStmt
        if (v->n4) v->n4->accept(*this); // RenameStmt
    }
    
    void Translator::visit(DeleteStmt *v) {
        // printf("<DeleteStmt>\n");
        stmtType_ = DELETE_STMT;
        if (v->n1) v->n1->accept(*this); // Table
        if (v->n2) v->n2->accept(*this); // Predicate
    }
    
    void Translator::visit(DmlStmt *v) {
        // printf("<DmlStmt>\n");
        if (v->n1) v->n1->accept(*this); // InsertStmt
        if (v->n2) v->n2->accept(*this); // SelectStmt
        if (v->n3) v->n3->accept(*this); // DeleteStmt
    }

    void Translator::visit(DropStmt *v) {
        // printf("<DropStmt>\n");
        stmtType_ = DROP_STMT;
        if (v->n1) v->n1->accept(*this); // Table
    }
    
    void Translator::visit(FromClause *v) {
        // printf("<FromClause>\n");
        if (v->n1)
            v->n1->accept(*this); // TableList
        else
            throw std::runtime_error("Unsupported SQL feature.");
        
    }
    
    void Translator::visit(InsertColumnList *v) {
        // printf("<InsertColumnList>\n");
        if (v->n1) v->n1->accept(*this); // InsertColumnList
        insertColumns_.push_back(v);
    }
    
    void Translator::visit(InsertStmt *v) {
        // printf("<InsertStmt>\n");
        stmtType_ = INSERT_STMT;
        if (v->n1) v->n1->accept(*this); // Table
        if (v->n2) v->n2->accept(*this); // InsertColumnList
        if (v->n3) v->n3->accept(*this); // LiteralList
    }
    
    void Translator::visit(Literal *v) {
        // printf("<Literal>\n");
        insertValues_.push_back(v);
        
        if (v->type == INT_TYPE)
            byteCount_ += sizeof(int);
        else if (v->type == FLOAT_TYPE)
            byteCount_ += sizeof(float);
        else
            throw std::runtime_error("Unsupported data type: " + std::to_string(v->type));
    }
    
    void Translator::visit(LiteralList *v) {
        // printf("<LiteralList>\n");
        if (v->n1) v->n1->accept(*this); // LiteralList
        if (v->n2) v->n2->accept(*this); // Literal
    }
    
    void Translator::visit(MapdDataT *v) {
        // printf("<MapdDataT>\n");
        createTypes_.push_back(v);
    }
    
    void Translator::visit(SQL_Namespace::MathExpr *v) {
        // printf("<MathExpr>\n");
        if (v->numericFlag)
            return;
        else if (v->n0)
            v->n0->accept(*this); // MathExpr
        else if (v->n1 && v->n2) {
            v->n1->accept(*this); // MathExpr
            v->n2->accept(*this); // MathExpr
        }
        else if (v->n3)
            v->n3->accept(*this); // Column
        else if (v->n4)
            v->n4->accept(*this); // AggrExpr
        else
            throw std::runtime_error("Unsupported SQL feature or badly formed math expression.");
    }
    
    
    void Translator::visit(OptAllDistinct *v) {
        // printf("<OptAllDistinct>\n");
        throw std::runtime_error("OptAllDistinct - unsupported SQL feature.");
    }
    
    void Translator::visit(OptGroupby *v) {
        // printf("<OptGroupby>\n");
        if (v->n1)
            v->n1->accept(*this); // ColumnList
        else
            throw std::runtime_error("OptGroupby - unsupported SQL feature.");
    }
    
    void Translator::visit(OptHaving *v) {
        // printf("<OptHaving>\n");
        throw std::runtime_error("OptHaving - unsupported SQL feature.");
    }
    
    void Translator::visit(OptLimit *v) {
        // printf("<OptLimit>\n");
        throw std::runtime_error("OptLimit - unsupported SQL feature.");
    }
    
    void Translator::visit(OptOrderby *v) {
        // printf("<OptOrderby>\n");
        throw std::runtime_error("OptOrderby - unsupported SQL feature.");
    }
    
    void Translator::visit(OptWhere *v) {
        // printf("<OptWhere>\n");
        if (v->n1) v->n1->accept(*this); // SearchCondition
    }
    
    void Translator::visit(SQL_Namespace::Predicate *v) {
        // printf("<Predicate>\n");
        queryInsidePredicate_ = true;
        if (v->op != "" && v->n1 && v->n2) {
            v->n1->accept(*this); // Predicate
            v->n2->accept(*this); // Predicate
        }
        else if (v->n3) {
            v->n3->accept(*this); // Comparison
        }
        else
            throw std::runtime_error("Predicate - unsupported SQL feature.");
        queryInsidePredicate_ = false;
    }
    
    void Translator::visit(RenameStmt *v) {
        // printf("<RenameStmt>\n");
        stmtType_ = RENAME_STMT;
        if (v->n1) v->n1->accept(*this); // Table
        renameTableNewName_ = v->name;
    }
    
    void Translator::visit(ScalarExpr *v) {
        // printf("<ScalarExpr>\n");
        
        if (v->n4)
            v->n4->accept(*this); // Column
        else
            throw std::runtime_error("Unsupported SQL feature.");
    }
    
    void Translator::visit(ScalarExprList *v) {
        // printf("<ScalarExprList>\n");
        if (v->n1) v->n1->accept(*this); // ScalarExprList
        if (v->n2) v->n2->accept(*this); // ScalarExpr
    }
    
    void Translator::visit(SearchCondition *v) {
        // printf("<SearchCondition>\n");
        if (stmtType_ == QUERY_STMT) {
            queryPredicate_ = v->n1;
            if (v->n1) v->n1->accept(*this); // Predicate
        }
        else
            throw std::runtime_error("Unsupported SQL feature.");
    }
    
    void Translator::visit(Selection *v) {
        // printf("<Selection>\n");
        
        if (v->all) {
            
            assert(stmtType_ == QUERY_STMT);
            assert(queryTables_.size() > 0);
            
            // perform an in-place translation of '*' in a selection clause
            // to a ScalarExprList of columns.
            std::vector<ColumnRow> columnsFromAllTables;
            for (size_t i = 0; i < queryTables_.size(); ++i)
                c_.getAllColumnMetadataForTable(queryTables_[i]->name.second, columnsFromAllTables);

            assert(columnsFromAllTables.size() > 0);
            
            ASTNode *subtree = new ScalarExprList(new ScalarExpr(new Column(columnsFromAllTables.back())));
            columnsFromAllTables.pop_back();
            ScalarExprList *tmpNode = nullptr;
            
            while (columnsFromAllTables.size() > 0) {
                tmpNode = (ScalarExprList*)subtree;
                tmpNode->n1 = new ScalarExprList(new ScalarExpr(new Column(columnsFromAllTables.back())));
                columnsFromAllTables.pop_back();
            }
            
            subtree->accept(*this);
            
        }
        else if (v->n1)
            v->n1->accept(*this); // ScalarExprList
    }

    void Translator::visit(SelectStmt *v) {
        // printf("<SelectStmt>\n");
        stmtType_ = QUERY_STMT;
        
        if (v->n1) v->n1->accept(*this); // OptAllDistinct

        // The FromClause must be visited before the Selection in order to
        // correct translate a "*" symbol in a Selection clause
        if (v->n3) v->n3->accept(*this); // FromClause
        if (v->n2) v->n2->accept(*this); // Selection
        
        if (v->n4) v->n4->accept(*this); // OptWhere
        if (v->n5) v->n5->accept(*this); // OptGroupby
        if (v->n6) v->n6->accept(*this); // OptHaving
        if (v->n7) v->n7->accept(*this); // OptOrderby
        if (v->n8) v->n8->accept(*this); // OptLimit
    }
    
    void Translator::visit(SqlStmt *v) {
        // printf("<SqlStmt>\n");
        if (v->n1) v->n1->accept(*this); // DmlStmt
        if (v->n2) v->n2->accept(*this); // DdlStmt
    }
    
    void Translator::visit(Table *v) {
        assert(v->name.first == ""); // @todo this should pass until we support NAME AS NAME
        // printf("<Table>\n");
        tableNames_.push_back(v->name.second);
        if (stmtType_ == QUERY_STMT)
            queryTables_.push_back(v);
        else if (stmtType_ == INSERT_STMT)
            insertTable_ = v;
        else if (stmtType_ == CREATE_STMT)
            createTableName_ = v;
        else if (stmtType_ == DROP_STMT)
            dropTableName_ = v;
        else if (stmtType_ == DELETE_STMT)
            deleteTableName_ = v;
        else if (stmtType_ == ALTER_STMT)
            ; // NOP; collected tableNames_ vector holds source table name
        else if (stmtType_ == RENAME_STMT)
            ; // NOP; collected tableNames_ vector holds source table name
        else
            throw std::runtime_error("Unsupported SQL statement.");
    }
    
    void Translator::visit(TableList *v) {
        // printf("<TableList>\n");
        if (v->n1) v->n1->accept(*this); // TableList
        if (v->n2) v->n2->accept(*this); // Table
    }

}
