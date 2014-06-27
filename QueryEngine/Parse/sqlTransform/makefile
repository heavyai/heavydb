sqlParser1Transform : lex.yy.c y.tab.c
	gcc -g lex.yy.c y.tab.c sqlParser1Graph.c transformer1.c -o sqlParser1Transform

y.tab.c : sqlParser1T.y
	yacc -y -d sqlParser1T.y -o y.tab.c

lex.yy.c : sqlParser1.l
	flex -i sqlParser1.l