#ifndef SQL_STATEMENT_H
#define SQL_STATEMENT_H

#include "ASTNode.h"

namespace SQL_Namespace { 

class Statement : public ASTNode {

public:
	virtual void accept(Visitor &v) = 0;
};

} // SQL_Namespace

#endif // SQL_STATEMENT_H
