#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>

#include "sqlParser1.h"
#include "y.tab.h"

/* Tree creator for the query "select * from table1" */
nodeType *firstNode; /* This should be an external pointer to the first node in the eventual tree. */
void checkNode(nodeType *p);
void handleSelectStatement(nodeType *queryNode);
nodeType *handle_Table_Exp(nodeType *queryNode);
nodeType *exploreTables(nodeType* table_list);
nodeType *opr(int oper, int nops, ...);

/******************************/

int transform(nodeType *p) {
	firstNode = p;
	checkNode(p);
	return ex(firstNode);
}

void checkNode(nodeType *p) {
	switch(p->type) { /*
        case typeCon: sprintf (word, "Float(%.1f)", p->con.dValue); break; 
        case typeId:  
            /* print at most first n characters (safe) */
            /*fprintf(stderr, "String to print: %s Characters to print: %d\n", p->id.s, p->id.iLength); 
            sprintf (word, "Identifier(%.*s)", p->id.iLength, p->id.s); break;
        case typeText: sprintf (word, "Text(%.*s)", p->id.iLength, p->id.s); break;
        case typeComp: sprintf (word, "Comparison(%.*s)", p->id.iLength, p->id.s); break;
        case typeAssgn: sprintf (word, "Assignment(%.*s)", p->id.iLength, p->id.s); break;
        */
        case typeOpr:
            switch(p->opr.oper) {
                case SQL:
                    /* This is where it all begins. Move to whatever the child is- could be manipulative statement or schema. We allow this parent to keep its children. For now. */
                	checkNode(p->opr.op[0]);
                	break;

                case MANIPULATIVE_STATEMENT:
                	/* Could be insert, update, or select. Let's go with select for now. */

                	checkNode(p->opr.op[0]); // Move to the only child.
                	break;

                case SELECT_STATEMENT:
				    /* Let's wri1te the skeleton for functions that will deal with each of these children rather
    				than make switch statements for every keyword. */

                	handle_Select_Statement(p);
                	break;
                }

                case SCHEMA:
                	/* could be create or drop table. Replace with BASE_TABLE_DEF */
                	checkNode(p->opr.op[0]);
                	break;

                case BASE_TABLE_DEF:
                	if (p->opr.op[0] == CREATE) handle_Create()

			}

}

void handle_Select_Statement(nodeType *queryNode) {
	/* Children are: SELECT keyword, opt_all_distinct which we are ignoring, selection, table_exp.*/
    firstNode = opr(PROJECT, 2, queryNode->opr.op[2]->opr.op[0], handle_Table_Exp(queryNode->opr.op[3]));
}

/* Return a Operator Node of type SELECT with child_0: operatorNode of type PREDICATE and 
child_1: operatorNode of type PRODUCT (reworking of Table_Ref_Commalist) */
nodeType *handle_Table_Exp(nodeType *queryNode) {
	/* Assuming the table_exp is intact, which is disturbing if it ain't, it should have five children. We care about the first two right now.

	1. From clause, with two children: 1. the word FROM, and the table_ref_commalist. We dissect the latter and return it as a node. 

	2. Where clause, with two children: 1. the word WHERE and 2. the search condition. 
	Powers that be suggest we are doing none of this for the first sprint, so we're leaving the structure the same- 
	it'll return [empty]. We're just going to assign the "search_condition" node to the   */
	
	nodeType *search_condition;

	/* check if opt_where_clause is empty o' not */
	if (queryNode->opr.op[1]->opr.nops == 1) {
		//if it's empty, make the node just [empty]
		search_condition = queryNode->opr.op[1]->opr.op[0];
	}
	else search_condition = queryNode->opr.op[1]->opr.op[1];


	nodeType *table_list = queryNode->opr.op[0]->opr.op[1];
	nodeType *finalProduct = exploreTables(table_list);

	// nodeType *whereClause = p->opr.op[1];
	return opr(SELECT, 2, search_condition, finalProduct);

}

nodeType *exploreTables(nodeType* table_list) {
	/* Check recursive base case. Return TABLE_REF_COMMALIST's child_0's child's child (see below comment) */
	if (table_list->opr.nops == 1) {
		return table_list->opr.op[0]->opr.op[0]->opr.op[0];
	}
	else {
		/* Assign to tableName TABLE_REF_COMMALIST's child_2's child's child,
		 or TABLE_REF's child's child, or TABLE's child, or the identifier(table_name) */
		nodeType *tableName = table_list->opr.op[2]->opr.op[0]->opr.op[0];
		return opr(PRODUCT, 2, exploreTables(table_list->opr.op[0]), tableName);
	}
}


/*
void freeNode(nodeType *p) {
    int i;

    if (!p) return;
    if (p->type == typeOpr) {
        for (i = 0; i < p->opr.nops; i++)
            freeNode(p->opr.op[i]);
		free (p->opr.op);
    }
    free (p);
}
*/
