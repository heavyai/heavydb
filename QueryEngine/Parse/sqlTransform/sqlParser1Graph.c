/* source code courtesy of Frank Thomas Braun */

/* calc3d.c: Generation of the graph of the syntax tree */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "sqlParser1.h"
#include "y.tab.h"

int del = 1; /* distance of graph columns */
int eps = 3; /* distance of graph lines */

/* interface for drawing (can be replaced by "real" graphic using GD or other) */
void graphInit (void);
void graphFinish();
void graphBox (char *s, int *w, int *h);
void graphDrawBox (char *s, int c, int l);
void graphDrawArrow (int c1, int l1, int c2, int l2);

/* recursive drawing of the syntax tree */
void exNode (nodeType *p, int c, int l, int *ce, int *cm);

/*****************************************************************************/

/* main entry point of the manipulation of the syntax tree */
int ex (nodeType *p) {
    int rte, rtm;

    graphInit ();
    exNode (p, 0, 0, &rte, &rtm);
    graphFinish();
    return 0;
}

/*c----cm---ce---->                       drawing of leaf-nodes
 l leaf-info
 */

/*c---------------cm--------------ce----> drawing of non-leaf-nodes
 l            node-info
 *                |
 *    -------------     ...----
 *    |       |               |
 *    v       v               v
 * child1  child2  ...     child-n
 *        che     che             che
 *cs      cs      cs              cs
 *
 */

void exNode
    (   nodeType *p,
        int c, int l,        /* start column and line of node */
        int *ce, int *cm     /* resulting end column and mid of node */
    )
{
    int w, h;           /* node width and height */
    char *s;            /* node text */
    int cbar;           /* "real" start column of node (centred above subnodes) */
    int k;              /* child number */
    int che, chm;       /* end column and mid of children */
    int cs;             /* start column of children */
    char word[20];      /* extended node text */

    if (!p) return;

    strcpy (word, "???"); /* should never appear */
    s = word;
    switch(p->type) {
        case typeCon: sprintf (word, "Float(%.1f)", p->con.dValue); break; 
        case typeId:  
            /* print at most first n characters (safe) */
            /*fprintf(stderr, "String to print: %s Characters to print: %d\n", p->id.s, p->id.iLength); */
            sprintf (word, "Identifier(%.*s)", p->id.iLength, p->id.s); break;
        case typeText: sprintf (word, "Text(%.*s)", p->id.iLength, p->id.s); break;
        case typeComp: sprintf (word, "Comparison(%.*s)", p->id.iLength, p->id.s); break;
        case typeAssgn: sprintf (word, "Assignment(%.*s)", p->id.iLength, p->id.s); break;

        case typeOpr:
            switch(p->opr.oper){
                case SELECT:     s = "SELECT"; break;
                case FROM:        s = "FROM";    break; 
                case DOT:     s = "[.]"; break; 
                case SELALL:       s = "[*]";     break; 
                case EMPTY:       s = "[empty]";     break;
                case WHERE:    s = "WHERE";     break;
                case OR:       s = "OR";     break; 
                case AND:       s = "AND";     break;
                case NOT:    s = "NOT";     break;
                case COMPARISON:    s = "COMPARISON";     break;
                case BETWEEN:    s = "BETWEEN";     break;
                case ALL:    s = "ALL";     break; 
                case DISTINCT:    s = "DISTINCT";     break;
                case HAVING:       s = "HAVING";     break;

                case UPDATE:    s = "UPDATE";     break;
                case SET:    s = "SET";     break;
                case OF:    s = "OF";     break; 
                case CURRENT:    s = "CURRENT";     break;
                case NULLX:       s = "NULL";     break;
                case ASSIGN:       s = "ASSIGN";     break;
                case ',':       s = "[,]";     break;
                case '.':       s = "[.]";     break; 
                case DOTNAME:       s = "DOTNAME";     break; 

                case INSERT:       s = "INSERT";     break;
                case VALUES:       s = "VALUES";     break;
                case INTO:       s = "INTO";     break;

                case CREATE:         s = "CREATE";      break;
                case DROP:         s = "DROP";      break;
                case TABLE:      s = "TABLE";       break;
                case UNIQUE:         s = "UNIQUE";      break;
                case PRIMARY:        s = "PRIMARY";     break;
                case FOREIGN:        s = "FOREIGN";     break;
                case KEY:        s = "KEY";     break;
                case CHECK:      s = "CHECK";       break;
                case REFERENCES:         s = "REFERENCES";      break;
                case DEFAULT:        s = "DEFAULT";     break;
                case USER:        s = "USER";     break;

                /* datatype cases */
                case DATATYPE:       s = "DATATYPE";     break;

                case DECIMAL:        s = "DECIMAL";     break;
                case SMALLINT:       s = "SMALLINT";        break;
                case NUMERIC:        s = "NUMERIC";     break;
                case CHARACTER:      s = "CHARACTER";       break;
                case INTEGER:        s = "INTEGER";     break;
                case REAL:             s = "REAL";        break;
                case FLOAT:         s = "FLOAT";       break;
                case DOUBLE:         s = "DOUBLE";      break;
                case PRECISION:      s = "PRECISION";       break;
                case VARCHAR:        s = "VARCHAR";     break;

                case AMMSC:       s = "AMMSC";        break;
                case AVG:       s = "AVG";        break;
                case MAX:      s = "MAX";       break;
                case MIN:         s = "MIN";      break;
                case SUM:      s = "SUM";       break;
                case COUNT:        s = "COUNT";     break;

                case GROUP:      s = "GROUP";       break;
                case ORDER:      s = "ORDER";       break;
                case BY:        s = "BY";     break;

                case AS:      s = "ALIAS";       break;
                case INTORDER:      s = "INTORDER";       break;
                case COLORDER:      s = "COLORDER";       break;
                case ASC:      s = "ASC";       break;
                case DESC:      s = "DESC";       break;
                case LIMIT:      s = "LIMIT";       break;
                case OFFSET:      s = "OFFSET";       break;

                /* excuse the mess. The following was generated by a python program. */
                case SQL:      s = "SQL";       break;
                case MANIPULATIVE_STATEMENT:      s = "MANIPULATIVE_STATEMENT";       break;
                case SELECT_STATEMENT:      s = "SELECT_STATEMENT";       break;
                case SELECTION:      s = "SELECTION";       break;
                case TABLE_EXP:      s = "TABLE_EXP";       break;
                case OPT_ALL_DISTINCT:      s = "OPT_ALL_DISTINCT";       break;
                case FROM_CLAUSE:      s = "FROM_CLAUSE";       break;
                case OPT_WHERE_CLAUSE:      s = "OPT_WHERE_CLAUSE";       break;
                case OPT_HAVING_CLAUSE:      s = "OPT_HAVING_CLAUSE";       break;
                case COLUMN_REF:      s = "COLUMN_REF";       break;
                case TABLE_REF_COMMALIST:      s = "TABLE_REF_COMMALIST";       break;
                case SCALAR_EXP_COMMALIST:      s = "SCALAR_EXP_COMMALIST";       break;
                case TABLE_REF:      s = "TABLE_REF";       break;
                case SEARCH_CONDITION:      s = "SEARCH_CONDITION";       break;
                case PREDICATE:      s = "PREDICATE";       break;
                case COMPARISON_PREDICATE:      s = "COMPARISON_PREDICATE";       break;
                case BETWEEN_PREDICATE:      s = "BETWEEN_PREDICATE";       break;
                case SCALAR_EXP:      s = "SCALAR_EXP";       break;
                case LITERAL:      s = "LITERAL";       break;
                case ATOM:      s = "ATOM";       break;
                case UPDATE_STATEMENT_POSITIONED:      s = "UPDATE_STATEMENT_POSITIONED";       break;
                case UPDATE_STATEMENT_SEARCHED:      s = "UPDATE_STATEMENT_SEARCHED";       break;
                case ASSIGNMENT:      s = "ASSIGNMENT";       break;
                case ASSIGNMENT_COMMALIST:      s = "ASSIGNMENT_COMMALIST";       break;
                case COLUMN:      s = "COLUMN";       break;
                case CURSOR:      s = "CURSOR";       break;
                case INSERT_STATEMENT:      s = "INSERT_STATEMENT";       break;
                case COLUMN_COMMALIST:      s = "COLUMN_COMMALIST";       break;
                case OPT_COLUMN_COMMALIST:      s = "OPT_COLUMN_COMMALIST";       break;
                case VALUES_OR_QUERY_SPEC:      s = "VALUES_OR_QUERY_SPEC";       break;
                case INSERT_ATOM_COMMALIST:      s = "INSERT_ATOM_COMMALIST";       break;
                case INSERT_ATOM:      s = "INSERT_ATOM";       break;
                case QUERY_SPEC:      s = "QUERY_SPEC";       break;
                case SCHEMA:      s = "SCHEMA";       break;
                case BASE_TABLE_DEF:      s = "BASE_TABLE_DEF";       break;
                case BASE_TABLE_ELEMENT_COMMALIST:      s = "BASE_TABLE_ELEMENT_COMMALIST";       break;
                case BASE_TABLE_ELEMENT:      s = "BASE_TABLE_ELEMENT";       break;
                case COLUMN_DEF:      s = "COLUMN_DEF";       break;
                case TABLE_CONSTRAINT_DEF:      s = "TABLE_CONSTRAINT_DEF";       break;
                case COLUMN_DEF_OPT:      s = "COLUMN_DEF_OPT";       break;
                case COLUMN_DEF_OPT_LIST:      s = "COLUMN_DEF_OPT_LIST";       break;
                case DATA_TYPE:      s = "DATA_TYPE";       break;
                case FUNCTION_REF:      s = "FUNCTION_REF";       break;
                case OPT_GROUP_BY_CLAUSE:      s = "OPT_GROUP_BY_CLAUSE";       break;
                case COLUMN_REF_COMMALIST:      s = "COLUMN_REF_COMMALIST";       break;
                case OPT_ASC_DESC:      s = "OPT_ASC_DESC";       break;
                case OPT_ORDER_BY_CLAUSE:      s = "OPT_ORDER_BY_CLAUSE";       break;
                case ORDERING_SPEC:      s = "ORDERING_SPEC";       break;
                case ORDERING_SPEC_COMMALIST:      s = "ORDERING_SPEC_COMMALIST";       break;
                case OPT_LIMIT_CLAUSE:      s = "OPT_LIMIT_CLAUSE";       break;

                case PRODUCT:      s = "PRODUCT";       break;
                case PROJECT:      s = "PROJECT";       break;
                /*case 'INSERT':       s = "[/]";     break;

                case '<':       s = "[<]";     break;
                case '>':       s = "[>]";     break;
                case '>=':        s = "[>=]";    break;
                case '<=':        s = "[<=]";    break;
                case '!=':        s = "[!=]";    break;
                case '=':        s = "[=]";    break; */
            }
            break;
    }

    /* construct node text box */
    graphBox (s, &w, &h);
    cbar = c;
    *ce = c + w;
    *cm = c + w / 2;

    /* node is leaf */
    if (p->type == typeCon || p->type == typeId || p->opr.nops == 0) {
        graphDrawBox (s, cbar, l);
        return;
    }

    /* node has children */
    cs = c;
    for (k = 0; k < p->opr.nops; k++) {
        exNode (p->opr.op[k], cs, l+h+eps, &che, &chm);
        cs = che;
    }

    /* total node width */
    if (w < che - c) {
        cbar += (che - c - w) / 2;
        *ce = che;
        *cm = (c + che) / 2;
    }

    /* draw node */
    graphDrawBox (s, cbar, l);

    /* draw arrows (not optimal: children are drawn a second time) */
    cs = c;
    for (k = 0; k < p->opr.nops; k++) {
        exNode (p->opr.op[k], cs, l+h+eps, &che, &chm);
        graphDrawArrow (*cm, l+h, chm, l+h+eps-1);
        cs = che;
    }
}

/* interface for drawing */

#define lmax 400
#define cmax 400

char graph[lmax][cmax]; /* array for ASCII-Graphic */
int graphNumber = 0;

void graphTest (int l, int c)
{   int ok;
    ok = 1;
    if (l < 0) ok = 0;
    if (l >= lmax) ok = 0;
    if (c < 0) ok = 0;
    if (c >= cmax) ok = 0;
    if (ok) return;
    printf ("\n+++error: l=%d, c=%d not in drawing rectangle 0, 0 ... %d, %d", 
        l, c, lmax, cmax);
    exit(1);
}

void graphInit (void) {
    int i, j;
    for (i = 0; i < lmax; i++) {
        for (j = 0; j < cmax; j++) {
            graph[i][j] = ' ';
        }
    }
}

void graphFinish() {
    int i, j;
    for (i = 0; i < lmax; i++) {
        for (j = cmax-1; j > 0 && graph[i][j] == ' '; j--);
        graph[i][cmax-1] = 0;
        if (j < cmax-1) graph[i][j+1] = 0;
        if (graph[i][j] == ' ') graph[i][j] = 0;
    }
    for (i = lmax-1; i > 0 && graph[i][0] == 0; i--);
    printf ("\n\nGraph %d:\n", graphNumber++);
    for (j = 0; j <= i; j++) printf ("\n%s", graph[j]);
    printf("\n");
}

void graphBox (char *s, int *w, int *h) {
    *w = strlen (s) + del;
    *h = 1;
}

void graphDrawBox (char *s, int c, int l) {
    int i;
    graphTest (l, c+strlen(s)-1+del);
    for (i = 0; i < strlen (s); i++) {
        graph[l][c+i+del] = s[i];
    }
}

void graphDrawArrow (int c1, int l1, int c2, int l2) {
    int m;
    graphTest (l1, c1);
    graphTest (l2, c2);
    m = (l1 + l2) / 2;
    while (l1 != m) { graph[l1][c1] = '|'; if (l1 < l2) l1++; else l1--; }
    while (c1 != c2) { graph[l1][c1] = '-'; if (c1 < c2) c1++; else c1--; }
    while (l1 != l2) { graph[l1][c1] = '|'; if (l1 < l2) l1++; else l1--; }
    graph[l1][c1] = '|';
}

