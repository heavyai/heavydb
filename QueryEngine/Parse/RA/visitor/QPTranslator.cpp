
/**
 * @file    QPTranslator.cpp
 * @author  Steven Stewart <steve@map-d.com>
 */
#include "QPTranslator.h"

#include "../ast/RelAlgNode.h"
#include "../ast/UnaryOp.h"
#include "../ast/BinaryOp.h"

#include "../ast/AggrExpr.h"
#include "../ast/AggrList.h"
#include "../ast/AntijoinOp.h"
#include "../ast/Attribute.h"
#include "../ast/AttrList.h"
#include "../ast/Comparison.h"
#include "../ast/DiffOp.h"
#include "../ast/Expr.h"
#include "../ast/ExtendOp.h"
#include "../ast/GroupbyOp.h"
#include "../ast/JoinOp.h"
#include "../ast/MathExpr.h"
#include "../ast/OuterjoinOp.h"
#include "../ast/Predicate.h"
#include "../ast/ProductOp.h"
#include "../ast/Program.h"
#include "../ast/ProjectOp.h"
#include "../ast/Relation.h"
#include "../ast/RelExpr.h"
#include "../ast/RelExprList.h"
#include "../ast/RenameOp.h"
#include "../ast/SelectOp.h"
#include "../ast/SemijoinOp.h"
#include "../ast/SortOp.h"
#include "../ast/UnionOp.h"

#include <iostream>
using std::cout;
using std::endl;

namespace RA_Namespace {

void QPTranslator::visit(class AggrExpr *v) {}
void QPTranslator::visit(class AggrList *v) {}
void QPTranslator::visit(class AntijoinOp *v) {}
void QPTranslator::visit(class Attribute *v) {}
void QPTranslator::visit(class AttrList *v) {}
void QPTranslator::visit(class Comparison *v) {}
void QPTranslator::visit(class DiffOp *v) {}
void QPTranslator::visit(class Expr *v) {}
void QPTranslator::visit(class ExtendOp *v) {}
void QPTranslator::visit(class GroupbyOp *v) {}
void QPTranslator::visit(class JoinOp *v) {}
void QPTranslator::visit(class MathExpr *v) {}
void QPTranslator::visit(class OuterjoinOp *v) {}
void QPTranslator::visit(class Predicate *v) {}
void QPTranslator::visit(class ProductOp *v) {}
void QPTranslator::visit(class Program *v) {}
void QPTranslator::visit(class ProjectOp *v) {}
void QPTranslator::visit(class Relation *v) {}
void QPTranslator::visit(class RelExpr *v) {}
void QPTranslator::visit(class RelExprList *v) {}
void QPTranslator::visit(class RenameOp *v) {}
void QPTranslator::visit(class SelectOp *v) {}
void QPTranslator::visit(class SemijoinOp *v) {}
void QPTranslator::visit(class SortOp *v) {}
void QPTranslator::visit(class UnionOp *v) {}

} // RA_Namespace
