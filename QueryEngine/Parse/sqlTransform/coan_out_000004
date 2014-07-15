typedef enum { typeCon, typeId, typeText, typeOpr, typeComp, typeAssgn } nodeEnum;

/* constants */
typedef struct {
	double dValue; /*value of constant */
	int iValue;
} conNodeType;

/* identifiers */
typedef struct {
		char *s;
		int iLength;
} idNodeType;

/* operators */
typedef struct {
	int oper;					/* operator */
	int nops;					/* no. of operands */
	struct nodeTypeTag **op; 	/* operands */
} oprNodeType;

typedef struct nodeTypeTag {
	nodeEnum type; 				/* type of node */

	union {
		conNodeType con; /* constants */
		idNodeType id; /* identifiers */
		oprNodeType opr; /* operators */
	};
} nodeType;

extern int textLength;
extern int textLength2; /* this is a backup in the case of NAME . NAME (see the grammar). 
						Both name strings correspond to a different pointer in the yytext string.
						Keeping both textLengths lets us make sure  */
extern int textLength3; /* For NAME.NAME.NAME */
extern int comparisonLength;

extern int dotFlag; /* to determine whether to assign length to textLength or textLength2. */