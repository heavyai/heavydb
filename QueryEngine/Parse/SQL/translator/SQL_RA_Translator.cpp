/**
 * @file	SQL_RA_Translator.cpp
 * @author	Steven Stewart <steve@map-d.com>
 */
#include <iostream>
#include "SQL_RA_Translator.h"

#include "../ast/DmlStmt.h"
#include "../ast/SqlStmt.h"

namespace SQL_Namespace {

RelAlgNode* SQL_RA_Translator::visit(DmlStmt* v) {
	
}

RelAlgNode* SQL_RA_Translator::visit(SqlStmt* v) {
	if (v->n1)
		return v->n1->accept(*this);
	else if (v->n2)
		return v->n2->accept(*this);
}

} // SQL_Namespace