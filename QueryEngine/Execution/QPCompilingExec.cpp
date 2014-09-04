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

QPCompilingExec::~QPCompilingExec() {
    module_ -> dump();
}

void QPCompilingExec::setupLlvm() {
    module_ = new llvm::Module("kernel",context_);
    llvm::FunctionType *funcType = llvm::FunctionType::get(builder_ -> getVoidTy(),false);
    llvm::Function *func = llvm::Function::Create(funcType,llvm::Function::ExternalLinkage, "func", module_);
    llvm::BasicBlock *entry = llvm::BasicBlock::Create(context_,"entrypoint",func);
    builder_ -> SetInsertPoint(entry);
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

void QPCompilingExec::matchOperands() {
    llvm::Value * right = valueStack_.top();
    valueStack_.pop();
    llvm::Value * left = valueStack_.top();
    valueStack_.pop();
    if (left->getType() != right->getType()) {
        //one is a float type
        if (left->getType()->isFloatTy()) {
            right = builder_->CreateSIToFP(right,llvm::Type::getFloatTy(context_),"conv");
        }
        else {
            left = builder_->CreateSIToFP(left,llvm::Type::getFloatTy(context_),"conv");
        }
    }
    valueStack_.push(left);
    valueStack_.push(right);
}


void QPCompilingExec::visit(MathExpr *v) {
	printf("<MathExpr>\n");
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
	if (v->n3) v->n3->accept(*this);
	if (v->n4) v->n4->accept(*this);
    if (v->isBinaryOp) {
        cout << "Is binary op" << endl;
        matchOperands(); // will ensure operands are either both int or both float
        llvm::Value * right = valueStack_.top();
        valueStack_.pop();
        llvm::Value * left = valueStack_.top();
        valueStack_.pop();
        bool operandsAreIntegers = left->getType()->isIntegerTy(); // will mean right is integer also 
        if (operandsAreIntegers) { 
            cout << "Ops are integers" << endl;
            switch (v->op) {
                case OP_ADD:
                    cout << "iadd" << endl;
                    valueStack_.push(builder_->CreateAdd(left,right,"add_tmp"));
                    break;
                case OP_SUBTRACT:
                    valueStack_.push(builder_->CreateSub(left,right,"sub_tmp"));
                    break;
                case OP_MULTIPLY:
                    cout << "imul" << endl;
                    valueStack_.push(builder_->CreateMul(left,right,"mul_tmp"));
                    break;
                case OP_DIVIDE:
                    valueStack_.push(builder_->CreateSDiv(left,right,"div_tmp"));
                    break;
            }
        }
        else { // floats
            cout << "Ops are floats" << endl;
            switch (v->op) {
                case OP_ADD:
                    cout << "fadd" << endl;
                    valueStack_.push(builder_->CreateFAdd(left,right,"add_tmp"));
                    break;
                case OP_SUBTRACT:
                    valueStack_.push(builder_->CreateFSub(left,right,"sub_tmp"));
                    break;
                case OP_MULTIPLY:
                    cout << "fmul" << endl;
                    valueStack_.push(builder_->CreateFMul(left,right,"mul_tmp"));
                    break;
                case OP_DIVIDE:
                    valueStack_.push(builder_->CreateFDiv(left,right,"div_tmp"));
                    break;
            }
        }
    }
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
