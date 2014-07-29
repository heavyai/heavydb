#ifndef SQL_STATEMENT_H
#define SQL_STATEMENT_H

#include "ASTNode.h"

class Statement : public ASTNode {

public:
	virtual void accept(Visitor &v) = 0;
};

#endif // SQL_STATEMENT_H
