#ifndef AST_VISITOR_H
#define AST_VISITOR_H

class Visitor {

public:
    /*
    // apparently this isn't supported, but it would be nice if it were:
    template <typename T>
    // virtual void visit(T &);*/

    virtual void visit(class Ammsc *v) = 0;
    virtual void visit(class Assignment *v) = 0;
    virtual void visit(class AssignmentCommalist *v) = 0;
    virtual void visit(class Atom *v) = 0;
    virtual void visit(class BaseTableDef *v) = 0;
    virtual void visit(class BaseTableElement *v) = 0;
    virtual void visit(class BaseTableElementCommalist *v) = 0;
    virtual void visit(class BetweenPredicate *v) = 0;
    virtual void visit(class Column *v) = 0;
    virtual void visit(class Cursor *v) = 0;
    virtual void visit(class ColumnCommalist *v) = 0;
    virtual void visit(class ColumnDef *v) = 0;
    virtual void visit(class ColumnDefOpt *v) = 0;
    virtual void visit(class ColumnDefOptList *v) = 0;
    virtual void visit(class ColumnRef *v) = 0;
    virtual void visit(class ColumnRefCommalist *v) = 0;
    virtual void visit(class ComparisonPredicate *v) = 0;
    virtual void visit(class DataType *v) = 0;
    virtual void visit(class FromClause *v) = 0;
    virtual void visit(class FunctionRef *v) = 0;
    virtual void visit(class InsertAtom *v) = 0;
    virtual void visit(class InsertAtomCommalist *v) = 0;
    virtual void visit(class InsertStatement *v) = 0;
    virtual void visit(class Literal *v) = 0;
    virtual void visit(class LikePredicate *v) = 0;
    virtual void visit(class ManipulativeStatement *v) = 0;
    virtual void visit(class OptAllDistinct *v) = 0;
    virtual void visit(class OptAscDesc *v) = 0;
    virtual void visit(class OptColumnCommalist *v) = 0;
    virtual void visit(class OptEscape *v) = 0;
    virtual void visit(class OptGroupByClause *v) = 0;
    virtual void visit(class OptHavingClause *v) = 0;
    virtual void visit(class OptLimitClause *v) = 0;
    virtual void visit(class OptOrderByClause *v) = 0;
    virtual void visit(class OptWhereClause *v) = 0;
    virtual void visit(class OrderingSpecCommalist *v) = 0;
    virtual void visit(class OrderingSpec *v) = 0;
    // virtual void visit(class ParameterRef *v) = 0;
    virtual void visit(class Predicate *v) = 0;
    virtual void visit (class Program *v) = 0;
    virtual void visit(class QuerySpec *v) = 0;
    virtual void visit(class SQL *v) = 0;
    virtual void visit(class SQLList *v) = 0;
    virtual void visit(class ScalarExp *v) = 0;
    virtual void visit(class ScalarExpCommalist *v) = 0;
    virtual void visit(class Schema *v) = 0;
    virtual void visit(class SearchCondition *v) = 0;
    virtual void visit(class SelectStatement *v) = 0;
    virtual void visit(class Selection *v) = 0;
    virtual void visit(class Table *v) = 0;
    virtual void visit(class TableConstraintDef *v) = 0;
    virtual void visit(class TableExp *v) = 0;
    virtual void visit(class TableRef *v) = 0;
    virtual void visit(class TableRefCommalist *v) = 0;
    virtual void visit(class UpdateStatementPositioned *v) = 0;
    virtual void visit(class UpdateStatementSearched *v) = 0;
    virtual void visit(class ValuesOrQuerySpec *v) = 0;
};

#endif // AST_VISITOR_H
