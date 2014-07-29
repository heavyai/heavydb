#ifndef SQL_STATEMENT_H
#define SQL_STATEMENT_H

class Statement : public ASTNode {

public:
	virtual void accept(Visitor &v) const = 0;
};

#endif // SQL_STATEMENT_H
