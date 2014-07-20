/**
 * @file	QPTranslator.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Gil Walzer <gil@map-d.com>
 *
 * This namespace contains functonality for the translation of SQL to a
 * query plan expressed in Map-D's intermediate representation, which
 * is a variation of relational algebra. The source for the SQL parser
 * is located in Parse/SQL, and the the AST nodes for creating the query
 * plan (which is a tree whose inner nodes are relational operators) are
 * located in Parse/RA.
 *
 * The QPTranslator class employs employs the visitor design pattern by
 * virtue of being derived from SQL/Visitor.h. It visits the SQL parse
 * tree in order to translate patterns (via well-specified translation
 * rules) into the query plan's language.
 */
#ifndef TRANSLATOR_H
#define TRANSLATOR_H

#include "../Parse/RA/ast/_RAHeaders.h"
#include "../Parse/SQL/ast/_SQLHeaders.h"
//#include "../Parse/SQL/visitor/Visitor.h"
//#include "../../Metadata/Catalog.h"

namespace Translator_Namespace {

/**
 * @class QPTranslator
 * @brief Visits an SQL parse tree in order to produce an RA parse tree
 */
/*class QPTranslator : public SQL_Namespace::Visitor {

public:
	/**
	 * @brief Returns the root node of the query plan tree
	 */
	/*RelAlgNode* getRoot() {
    	return root;
	}

	void visit(ASTNode *v) {
		; // NOOP;
	}

	void visit(SelectStatement *v) {
		
	}

};*/

} // Translator_Namespace

#endif