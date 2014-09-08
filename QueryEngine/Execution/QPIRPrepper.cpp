/**
 * @file	QPIRPrepper.cpp
 * @author	Todd Mostak <todd@map-d.com>
 *
 * Implementation of RA query plan walker/compiler.
 */
#include "QPIRPrepper.h"

#include "../Parse/RA/ast/AggrExpr.h"
#include "../Parse/RA/ast/Attribute.h"
#include "../Parse/RA/ast/AttrList.h"
#include "../Parse/RA/ast/BinaryOp.h"
#include "../Parse/RA/ast/Comparison.h"
#include "../Parse/RA/ast/MathExpr.h"
#include "../Parse/RA/ast/Predicate.h"
#include "../Parse/RA/ast/Program.h"
#include "../Parse/RA/ast/ProjectOp.h"
#include "../Parse/RA/ast/Relation.h"
#include "../Parse/RA/ast/RelExpr.h"
#include "../Parse/RA/ast/RelExprList.h"
#include "../Parse/RA/ast/SelectOp.h"
#include "../Parse/RA/ast/UnaryOp.h"

#include <iostream>
#include <utility>
using namespace std;

namespace Execution_Namespace {

QPIRPrepper::~QPIRPrepper() {
    //cout << "QPIRPrepper:\n" << signatureString_ << endl;  
}

void QPIRPrepper::visit(Attribute *v) {
    signatureString_ += "<Attribute>";
    //@todo temporary for testing
    signatureString_ += "<Int Attr>";
    attributeNodes_.push_back(v);
    //@todo should be some type of function
    /*
    switch(v->metadata->columnType) {
        case(INT_TYPE):
            signatureString_ += "<Int Attr>;
            break;
        case(FLOAT_TYPE):
            signatureString_ += "<Float Attr>;
            break;
    }
    */
    signatureString_ += "</Attribute>";
}

void QPIRPrepper::visit(AggrExpr *v) {
    signatureString_ += "<AggrExpr>";
	if (v->n1) v->n1->accept(*this);
    signatureString_ += "</AggrExpr>";
}

void QPIRPrepper::visit(AttrList *v) {
    signatureString_ += "<AttrList>";
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
    signatureString_ += "</AttrList>";
}

void QPIRPrepper::visit(Comparison *v) {
    signatureString_ += "<Comparison>";
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
    switch (v->op) {
        case OP_GT:
            signatureString_ += "<OP_GT>";
            break;
        case OP_LT:
            signatureString_ += "<OP_LT>";
            break;
        case OP_GTE:
            signatureString_ += "<OP_GTE>";
            break;
        case OP_LTE:
            signatureString_ += "<OP_LTE>";
            break;
        case OP_NEQ:
            signatureString_ += "<OP_NEQ>";
            break;
        case OP_EQ:
            signatureString_ += "<OP_EQ>";
            break;
    }
    signatureString_ += "</Comparison>";
}

void QPIRPrepper::visit(MathExpr *v) {
    signatureString_ += "<MathExpr>";
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
	if (v->n3) v->n3->accept(*this);
	if (v->n4) v->n4->accept(*this);
    if (v->isBinaryOp) {
        switch (v->op) {
            case OP_ADD:
                signatureString_ += "<OP_ADD>";
                break;
            case OP_SUBTRACT:
                signatureString_ += "<OP_SUBTRACT>";
                break;
            case OP_MULTIPLY:
                signatureString_ += "<OP_MULTIPLY>";
                break;
            case OP_DIVIDE:
                signatureString_ += "<OP_DIVIDE>";
                break;
        }
    }
    else if (v->isScalar) {
        if (v->intFloatFlag) {
            signatureString_ += "<IntScalar>";
        }
        else {
            signatureString_ += "<FloatScalar>";
        }
        constantNodes_.push_back(v);
    }
    signatureString_ += "</MathExpr>";
}

void QPIRPrepper::visit(Predicate *v) {
	signatureString_ += "<Predicate>";
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
	if (v->n3) v->n3->accept(*this);
	signatureString_ += "</Predicate>";
}

void QPIRPrepper::visit(Program *v) {
	signatureString_ += "<Program>";
	if (v->n1) v->n1->accept(*this);
	signatureString_ += "</Program>";
}

void QPIRPrepper::visit(ProjectOp *v) {
	signatureString_ += "<ProjectOp>";
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
	signatureString_ += "</ProjectOp>";
}

void QPIRPrepper::visit(Relation *v) {
	signatureString_ += "<Relation>";
	signatureString_ += "</Relation>";
}

void QPIRPrepper::visit(RelExpr *v) {
	signatureString_ += "<RelExpr>";
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
	if (v->n3) v->n3->accept(*this);
	if (v->n4) v->n4->accept(*this);
	signatureString_ += "</RelExpr>";
}

void QPIRPrepper::visit(RelExprList *v) {
	signatureString_ += "<RelExprList>";
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
	signatureString_ += "</RelExprList>";
}

void QPIRPrepper::visit(SelectOp *v) {
	signatureString_ += "<SelectOp>";
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
	signatureString_ += "</SelectOp>";
}

} // Execution_Namespace
