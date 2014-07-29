/**
 * @file    Predicate.h
 * @author  Steven Stewart <steve@map-d.com>
 */
#ifndef SQL_PREDICATE_NODE_H
#define SQL_PREDICATE_NODE_H

#include <cassert>
#include "ASTNode.h"
#include "../visitor/Visitor.h"

class Predicate : public ASTNode {
    
public:
	Predicate *n1 = NULL;
	Predicate *n2 = NULL;
	Comparison *n3 = NULL;
    std::string op = "";

    /// Constructor
    Predicate(const std::string &op, Predicate *n1, Predicate *n2) {
    	assert(op == "AND" || op == "OR");
        assert(n1 && n2);
    	this->op = op;
    	this->n1 = n1;
    	this->n2 = n2;
        this->n3 = NULL;
    }

    Predicate(const std::string &op, Predicate *n1) {
    	assert(op == "NOT");
        assert(n1);
    	this->op = op;
    	this->n1 = n1;
        this->n2 = NULL;
        this->n3 = NULL;
    }

    explicit Predicate(Predicate *n1) {
        assert(n1);
        this->op = "";
    	this->n1 = n1;
        this->n2 = NULL;
        this->n3 = NULL;
    }

    explicit Predicate(Comparison *n3) {
        assert(n3);
        this->op = "";
        this->n1 = NULL;
        this->n2 = NULL;
    	this->n3 = n3;
    }

	virtual void accept(class Visitor &v) const {
		v.visit(this);
	}
};

#endif // SQL_PREDICATE_NODE_H
