/**
 * @file	Ammsc.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Gil Walzer <gil@map-d.com>
 */
#ifndef SQL_AST_AMMSC_H
#define SQL_AST_AMMSC_H

#include <cassert>
#include <string>
#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {

class Ammsc : public ASTNode {
    
public:
    std::string funcName = "";
    
    /// Constructor
    Ammsc(const std::string &n1) {
    	assert(n1 == std::string("COUNT") || n1 == std::string("AVG") || n1 == std::string("MAX") || n1 == std::string("MIN") || n1 == std::string("SUM"));
    	funcName = n1;
    }

	virtual void accept(class Visitor &v) {
		v.visit(this);
	}
};

} // SQL_Namespace

#endif // SQL_AST_AMMSC_H
