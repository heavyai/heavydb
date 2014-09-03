/**
 * @file	QPCompilingExec.cpp
 * @author	Todd Mostak <steve@map-d.com>
 *
 * Implementation of RA query plan walker/compiler.
 */
#include "QPCompilingExec.h"

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

QPCompilingExec::QPCompilingExec(): context_(llvm::getGlobalContext()), builder_(new llvm::IRBuilder <> (context_)) {
    setupLlvm();
} 

void QPCompilingExec::setupLlvm() {
    module_ = new llvm::Module("kernel",context_);
    llvm::FunctionType *funcType = llvm::FunctionType::get(builder_ -> getVoidTy(),false);
    llvm::Function *func = llvm::Function::Create(funcType,llvm::Function::ExternalLinkage, "func", module_);
    llvm::BasicBlock *entry = llvm::BasicBlock::Create(context_,"entrypoint",func);
    builder_ -> SetInsertPoint(entry);
    module_ -> dump();
}

void QPCompilingExec::visit(Attribute *v) {
	printf("<Attribute>\n");
}

void QPCompilingExec::visit(AggrExpr *v) {
	printf("<AggrExpr>\n");
	if (v->n1) v->n1->accept(*this);
}

void QPCompilingExec::visit(AttrList *v) {
	printf("<AttrList>\n");
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
}

void QPCompilingExec::visit(Comparison *v) {
	printf("<Comparison>\n");
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
}

void QPCompilingExec::visit(MathExpr *v) {
	printf("<MathExpr>\n");
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
	if (v->n3) v->n3->accept(*this);
	if (v->n4) v->n4->accept(*this);
    /*
    if (v->isBinaryOp) {
        Visitor * right = valueStack_.top();
        Visitor * left = valueStack_.top();
        valueStack_.pop();
        valueStack_.pop();
        switch (v->op) {
            case OP_PLUS:
                valueStack_.push(builder_->createFAdd(left,rihgt,"addtmp"));
                break;
            case OP_MINUS:
                valueStack_.push(builder_->createFSub(left,rihgt,"addtmp"));

        else if (v->op == "MINUS") {
        




    }
    */
    else if (v->isScalar) {
        if (v->intFloatFlag) {
            cout << "Int: " << v->intVal << endl;
            valueStack_.push(llvm::ConstantInt::get(context_,llvm::APInt(32,v->intVal,true)));
        }
        else {
            cout << "Float: " << v->floatVal << endl;
            valueStack_.push(llvm::ConstantFP::get(context_,llvm::APFloat(v->floatVal)));
        }
    }

}

void QPCompilingExec::visit(Predicate *v) {
	printf("<Predicate>\n");
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
	if (v->n3) v->n3->accept(*this);
}

void QPCompilingExec::visit(Program *v) {
	printf("<Program>\n");
	if (v->n1) v->n1->accept(*this);
}

void QPCompilingExec::visit(ProjectOp *v) {
	printf("<ProjectOp>\n");
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
}

void QPCompilingExec::visit(Relation *v) {
	printf("<Relation>\n");
}

void QPCompilingExec::visit(RelExpr *v) {
	printf("<RelExpr>\n");
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
	if (v->n3) v->n3->accept(*this);
	if (v->n4) v->n4->accept(*this);
}

void QPCompilingExec::visit(RelExprList *v) {
	printf("<RelExprList>\n");
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
}

void QPCompilingExec::visit(SelectOp *v) {
	printf("<SelectOp>\n");
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
}

} // Execution_Namespace
