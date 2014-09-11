/**
 * @file	QPIRPrepper.h
 * @author	Todd Mostak <todd@map-d.com>
 */
#ifndef QUERYENGINE_EXECUTION_QPIRPREPPER_H
#define QUERYENGINE_EXECUTION_QPIRPREPPER_H


#include "../../Shared/types.h"
#include "../Parse/RA/visitor/Visitor.h"

#include <iostream>
#include <vector>
#include <map>


//#include "../../DataMgr/Metadata/Catalog.h"

using namespace RA_Namespace;

namespace Execution_Namespace {

/**
 * @class 	QPStringifyTree
 * @brief	This class walks/compiles a query plan, which is an AST of relational algebra statements.
 */
class QPIRPrepper : public Visitor {

public:
	/// Constructor
	QPIRPrepper(); 
    ~QPIRPrepper();
	
	/// Returns an error message if an error was encountered
	inline std::pair<bool, std::string> isError() { return std::pair<bool, std::string>(errFlag_, errMsg_); }

	/// @brief Visit a AggrExpr node
	virtual void visit(AggrExpr *v);

	/// @brief Visit a Attribute node
	virtual void visit(Attribute *v);

	/// @brief Visit a AttrList node
	virtual void visit(AttrList *v);

	/// @brief Visit a Comparison node
	virtual void visit(Comparison *v);

	/// @brief Visit a MathExpr node
	virtual void visit(MathExpr *v);

	/// @brief Visit a Predicate node
	virtual void visit(Predicate *v);

	/// @brief Visit a Program node
	virtual void visit(Program *v);

	/// @brief Visit a ProjectOp node
	virtual void visit(ProjectOp *v);

	/// @brief Visit a Relation node
	virtual void visit(Relation *v);

	/// @brief Visit a RelExpr node
	virtual void visit(RelExpr *v);
	
	/// @brief Visit a RelExprList node
	virtual void visit(RelExprList *v);

	/// @brief Visit a SelectOp node
	virtual void visit(SelectOp *v);

    inline void getSignatureString(std::string &sigString) {
        sigString = signatureString_;
    }

    std::vector <Attribute *> projectNodes_;
    std::vector <Attribute *> attributeNodes_;
    std::vector <MathExpr *> constantNodes_;

private:

    std::string signatureString_;
	std::string errMsg_;	/// holds an error message, if applicable; otherwise, it is ""
	bool errFlag_ = false;	/// indicates the existence of an error when true
    bool inProject_;

};

} // Execution_Namespace

#endif // QUERYENGINE_EXECUTION_QPIRPREPPER_H
