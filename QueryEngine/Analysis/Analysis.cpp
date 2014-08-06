/**
 * @file	Analysis.cpp
 * @author	Steven Stewart <steve@map-d.com>
 * @brief	Implementation of semantic analysis functions specified in Analysis.h
 */
#include "Analysis.h"
#include "../Parse/SQL/ast/ASTNode.h"
#include "../../DataMgr/Metadata/Catalog.h"

namespace Analysis_Namespace {

std::pair<bool, std::string> checkInsert(ASTNode* tree, Catalog& c, InsertData& insert) {
	InsertWalker w(c);
	tree->accept(w);
	return (w.isError());
}

std::pair<bool, std::string> checkSql(ASTNode *tree, Catalog &c) {
	TypeChecker w(c);
	tree->accept(w);
	return (w.isError());
}

} // Analysis_Namespace
