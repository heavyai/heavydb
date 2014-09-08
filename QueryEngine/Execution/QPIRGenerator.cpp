/**
 * @file	QPIRGenerator.cpp
 * @author	Todd Mostak <todd@map-d.com>
 * @author	Steve Stewart <steve@map-d.com>
 *
 * Implementation of RA query plan walker/compiler.
 */
#include "QPIRGenerator.h"

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

QPIRGenerator::QPIRGenerator(vector<Attribute *> &attributeNodes, vector<MathExpr *> &constantNodes): attributeNodes_(attributeNodes), constantNodes_(constantNodes), context_(llvm::getGlobalContext()), builder_(new llvm::IRBuilder <> (context_)) {
    setupLlvm();
} 

QPIRGenerator::~QPIRGenerator() {
    //module_ -> dump();
}

void QPIRGenerator::setupLlvm() {
    module_ = new llvm::Module("kernel",context_);
    //vector<Type *> argumentTypes(attributeNodes_.size() + constantNodes_.size());
    /*
    vector<Type *> argumentTypes;
    for (auto attrIt = attributeNodes_.begin(); attrIt != attributeNodes_.end(); ++attrIt) {
        argumentTypes.push_back(Type::getFloatTY(context_));
    }
    for (auto constIt = constantNodes_.begin(); constIt != constantNodes_.end(); ++constIt) {
        if (*(constIt) -> intFloatFlag) { // is int
            argumentTypes.push_back(Type::

        switch(*(constIt)
    */
    

    llvm::FunctionType *funcType = llvm::FunctionType::get(builder_ -> getVoidTy(),false);
    llvm::Function *func = llvm::Function::Create(funcType,llvm::Function::ExternalLinkage, "func", module_);
    llvm::BasicBlock *entry = llvm::BasicBlock::Create(context_,"entrypoint",func);
    /*
    llvm::BasicBlock *entry = llvm::BasicBlock::Create(context_,"entrypoint");
    */
    builder_ -> SetInsertPoint(entry);

}
void QPIRGenerator::visit(Attribute *v) {
    /*
	printf("<Attribute>\n");
    auto varMapIt = varMap_.find(v->name1);
    if (varMapIt != varMap_.end()) { // var already exists in map
        valueStack_.push(varMapIt -> second);  
    }
    else { // first time we've seen this variable
        //@todo - get attribute type - for now just assume float
        Value *value =  
         
    }
    */
}

void QPIRGenerator::visit(AggrExpr *v) {
	printf("<AggrExpr>\n");
	if (v->n1) v->n1->accept(*this);
}

void QPIRGenerator::visit(AttrList *v) {
	printf("<AttrList>\n");
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
}

void QPIRGenerator::visit(Comparison *v) {
	printf("<Comparison>\n");
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
    matchOperands();
    llvm::Value * right = valueStack_.top();
    valueStack_.pop();
    llvm::Value * left = valueStack_.top();
    valueStack_.pop();
    bool operandsAreIntegers = left->getType()->isIntegerTy(); // will mean right is integer also 
    if (operandsAreIntegers) { 
        switch (v->op) {
            case OP_GT:
                cout << "i-gt" << endl;
                valueStack_.push(builder_->CreateICmpSGT(left,right,"cmpigt"));
                break;
            case OP_LT:
                cout << "i-lt" << endl;
                valueStack_.push(builder_->CreateICmpSLT(left,right,"cmpilt"));
                break;
            case OP_GTE:
                cout << "i-gte" << endl;
                valueStack_.push(builder_->CreateICmpSGE(left,right,"cmpigte"));
                break;
            case OP_LTE:
                cout << "i-lte" << endl;
                valueStack_.push(builder_->CreateICmpSLE(left,right,"cmpilte"));
                break;
            case OP_NEQ:
                cout << "i-neq" << endl;
                valueStack_.push(builder_->CreateICmpNE(left,right,"cmpineq"));
                break;
            case OP_EQ:
                cout << "i-eq" << endl;
                valueStack_.push(builder_->CreateICmpEQ(left,right,"cmpieq"));
                break;
        }
    }
    else {
        switch (v->op) {
            case OP_GT:
                cout << "f-gt" << endl;
                valueStack_.push(builder_->CreateFCmpUGT(left,right,"cmpfgt"));
                break;
            case OP_LT:
                cout << "f-lt" << endl;
                valueStack_.push(builder_->CreateFCmpULT(left,right,"cmpflt"));
                break;
            case OP_GTE:
                cout << "f-gte" << endl;
                valueStack_.push(builder_->CreateFCmpUGE(left,right,"cmpfgte"));
                break;
            case OP_LTE:
                cout << "f-lte" << endl;
                valueStack_.push(builder_->CreateFCmpULE(left,right,"cmpflte"));
                break;
            case OP_NEQ:
                cout << "f-neq" << endl;
                valueStack_.push(builder_->CreateFCmpUNE(left,right,"cmpfneq"));
                break;
            case OP_EQ:
                cout << "f-eq" << endl;
                valueStack_.push(builder_->CreateFCmpUEQ(left,right,"cmpfeq"));
                break;
        }

    }

}

void QPIRGenerator::matchOperands() {
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


void QPIRGenerator::visit(MathExpr *v) {
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

void QPIRGenerator::visit(Predicate *v) {
	printf("<Predicate>\n");
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
	if (v->n3) v->n3->accept(*this);
}

void QPIRGenerator::visit(Program *v) {
	printf("<Program>\n");
	if (v->n1) v->n1->accept(*this);
    builder_ -> CreateRetVoid();
}

void QPIRGenerator::visit(ProjectOp *v) {
	printf("<ProjectOp>\n");
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
}

void QPIRGenerator::visit(Relation *v) {
	printf("<Relation>\n");
}

void QPIRGenerator::visit(RelExpr *v) {
	printf("<RelExpr>\n");
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
	if (v->n3) v->n3->accept(*this);
	if (v->n4) v->n4->accept(*this);
}

void QPIRGenerator::visit(RelExprList *v) {
	printf("<RelExprList>\n");
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
}

void QPIRGenerator::visit(SelectOp *v) {
	printf("<SelectOp>\n");
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
}

} // Execution_Namespace
