sqlParser1AlteredInput : lex.yy.c y.tab.c
	g++ -g lex.yy.c y.tab.c parseThis.c -o sqlParser1AlteredInput

y.tab.c : sqlParser1AlteredInput.y
	yacc -y -d sqlParser1AlteredInput.y -o y.tab.c

lex.yy.c : sqlParser1AlteredInput.l
	flex -i sqlParser1AlteredInput.l