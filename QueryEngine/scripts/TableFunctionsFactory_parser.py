__all__ = ['Parser']

import sys
import TableFunctionsFactory_node as tf_node
from collections import deque

if sys.version_info > (3, 0):
    from collections.abc import Iterable
else:
    from collections import Iterable


class TokenizeException(Exception):
    pass


class ParserException(Exception):
    pass


class Token:
    LESS = 1         # <
    GREATER = 2      # >
    COMMA = 3        # ,
    EQUAL = 4        # =
    RARROW = 5       # ->
    STRING = 6       # reserved for string constants
    NUMBER = 7       #
    VBAR = 8         # |
    BANG = 9         # !
    LPAR = 10        # (
    RPAR = 11        # )
    LSQB = 12        # [
    RSQB = 13        # ]
    IDENTIFIER = 14  #
    COLON = 15       # :
    BOOLEAN = 16     #

    def __init__(self, type, lexeme):
        """
        Parameters
        ----------
        type : int
          One of the tokens in the list above
        lexeme : str
          Corresponding string in the text
        """
        self.type = type
        self.lexeme = lexeme

    @classmethod
    def tok_name(cls, token):
        names = {
            Token.LESS: "LESS",
            Token.GREATER: "GREATER",
            Token.COMMA: "COMMA",
            Token.EQUAL: "EQUAL",
            Token.RARROW: "RARROW",
            Token.STRING: "STRING",
            Token.NUMBER: "NUMBER",
            Token.VBAR: "VBAR",
            Token.BANG: "BANG",
            Token.LPAR: "LPAR",
            Token.RPAR: "RPAR",
            Token.LSQB: "LSQB",
            Token.RSQB: "RSQB",
            Token.IDENTIFIER: "IDENTIFIER",
            Token.COLON: "COLON",
            Token.BOOLEAN: "BOOLEAN"
        }
        return names.get(token)

    def __str__(self):
        return 'Token(%s, "%s")' % (Token.tok_name(self.type), self.lexeme)

    __repr__ = __str__


class Tokenize:
    def __init__(self, line):
        self._line = line
        self._tokens = []
        self.start = 0
        self.curr = 0
        self.tokenize()

    @property
    def line(self):
        return self._line

    @property
    def tokens(self):
        return self._tokens

    def tokenize(self):
        while not self.is_at_end():
            self.start = self.curr

            if self.is_token_whitespace():
                self.consume_whitespace()
            elif self.is_number():
                self.consume_number()
            elif self.is_token_string():
                self.consume_string()
            elif self.is_token_identifier_or_boolean():
                self.consume_identifier_or_boolean()
            elif self.can_token_be_double_char():
                self.consume_double_char()
            else:
                self.consume_single_char()

    def is_at_end(self):
        return len(self.line) == self.curr

    def current_token(self):
        return self.line[self.start:self.curr + 1]

    def add_token(self, type):
        lexeme = self.line[self.start:self.curr + 1]
        self._tokens.append(Token(type, lexeme))

    def lookahead(self):
        if self.curr + 1 >= len(self.line):
            return None
        return self.line[self.curr + 1]

    def advance(self):
        self.curr += 1

    def peek(self):
        return self.line[self.curr]

    def can_token_be_double_char(self):
        char = self.peek()
        return char in ("-",)

    def consume_double_char(self):
        ahead = self.lookahead()
        if ahead == ">":
            self.advance()
            self.add_token(Token.RARROW)  # ->
            self.advance()
        else:
            self.raise_tokenize_error()

    def consume_single_char(self):
        char = self.peek()
        if char == "(":
            self.add_token(Token.LPAR)
        elif char == ")":
            self.add_token(Token.RPAR)
        elif char == "<":
            self.add_token(Token.LESS)
        elif char == ">":
            self.add_token(Token.GREATER)
        elif char == ",":
            self.add_token(Token.COMMA)
        elif char == "=":
            self.add_token(Token.EQUAL)
        elif char == "|":
            self.add_token(Token.VBAR)
        elif char == "!":
            self.add_token(Token.BANG)
        elif char == "[":
            self.add_token(Token.LSQB)
        elif char == "]":
            self.add_token(Token.RSQB)
        elif char == ":":
            self.add_token(Token.COLON)
        else:
            self.raise_tokenize_error()
        self.advance()

    def consume_whitespace(self):
        self.advance()

    def consume_string(self):
        """
        STRING: \".*?\"
        """
        while True:
            char = self.lookahead()
            curr = self.peek()
            if char == '"' and curr != '\\':
                self.advance()
                break
            self.advance()
        self.add_token(Token.STRING)
        self.advance()

    def consume_number(self):
        """
        NUMBER: [-]([0-9]*[.])?[0-9]+
        """
        found_dot = False
        while True:
            char = self.lookahead()
            if char:
                if char.isdigit():
                    self.advance()
                elif char == "." and not found_dot:
                    found_dot = True
                    self.advance()
                else:
                    break
            else:
                break
        self.add_token(Token.NUMBER)
        self.advance()

    def consume_identifier_or_boolean(self):
        """
        IDENTIFIER: [A-Za-z_][A-Za-z0-9_]*
        """
        while True:
            char = self.lookahead()
            if char and char.isalnum() or char == "_":
                self.advance()
            else:
                break
        if self.current_token().lower() in ("true", "false"):
            self.add_token(Token.BOOLEAN)
        else:
            self.add_token(Token.IDENTIFIER)
        self.advance()

    def is_token_identifier_or_boolean(self):
        return self.peek().isalpha() or self.peek() == "_"

    def is_token_string(self):
        return self.peek() == '"'

    def is_number(self):
        return self.peek().isdigit() or (self.peek() == '-' \
            and self.lookahead().isdigit())

    def is_alpha(self):
        return self.peek().isalpha()

    def is_token_whitespace(self):
        return self.peek().isspace()

    def raise_tokenize_error(self):
        curr = self.curr
        char = self.peek()
        raise TokenizeException(
            'Could not match char "%s" at pos %d on line\n  %s' % (char, curr, self.line)
        )


def is_identifier_cursor(identifier):
    return identifier.lower() == 'cursor'


class Parser:
    def __init__(self, line):
        self._tokens = Tokenize(line).tokens
        self._curr = 0
        self.line = line

    @property
    def tokens(self):
        return self._tokens

    def is_at_end(self):
        return self._curr >= len(self._tokens)

    def current_token(self):
        return self._tokens[self._curr]

    def advance(self):
        self._curr += 1

    def expect(self, expected_type):
        curr_token = self.current_token()
        msg = "Expected token %s but got %s at pos %d.\n Tokens: %s" % (
            curr_token,
            Token.tok_name(expected_type),
            self._curr,
            self._tokens,
        )
        assert curr_token.type == expected_type, msg
        self.advance()

    def consume(self, expected_type):
        """consumes the current token iff its type matches the
        expected_type. Otherwise, an error is raised
        """
        curr_token = self.current_token()
        if curr_token.type == expected_type:
            self.advance()
            return curr_token
        else:
            expected_token = Token.tok_name(expected_type)
            self.raise_parser_error(
                'Token mismatch at function consume. '
                'Expected type "%s" but got token "%s"\n\n'
                'Tokens: %s\n' % (expected_token, curr_token, self._tokens)
            )

    def current_pos(self):
        return self._curr

    def raise_parser_error(self, msg=None):
        if not msg:
            token = self.current_token()
            pos = self.current_pos()
            tokens = self.tokens
            msg = "\n\nError while trying to parse token %s at pos %d.\n" "Tokens: %s" % (
                token,
                pos,
                tokens,
            )
        raise ParserException(msg)

    def match(self, expected_type):
        curr_token = self.current_token()
        return curr_token.type == expected_type

    def lookahead(self):
        return self._tokens[self._curr + 1]

    def parse_udtf(self):
        """fmt: off

        udtf: IDENTIFIER "(" (args)? ")" ("|" annotation)* "->" args ("," templates)? ("|" "output_row_size" "=" primitive)?

        fmt: on
        """
        name = self.parse_identifier()
        self.expect(Token.LPAR)  # (
        input_args = []
        if not self.match(Token.RPAR):
            input_args = self.parse_args()
        self.expect(Token.RPAR)  # )
        annotations = []
        while not self.is_at_end() and self.match(Token.VBAR):  # |
            self.consume(Token.VBAR)
            annotations.append(self.parse_annotation())
        self.expect(Token.RARROW)  # ->
        output_args = self.parse_args()

        templates = None
        if not self.is_at_end() and self.match(Token.COMMA):
            self.consume(Token.COMMA)
            templates = self.parse_templates()

        sizer = None
        if not self.is_at_end() and self.match(Token.VBAR):
            self.consume(Token.VBAR)
            idtn = self.parse_identifier()
            assert idtn == "output_row_size", idtn
            self.consume(Token.EQUAL)
            node = self.parse_primitive()
            key = "kPreFlightParameter"
            sizer = tf_node.AnnotationNode(key, value=node.type)

        # set arg_pos
        i = 0
        for arg in input_args:
            arg.arg_pos = i
            arg.kind = "input"
            i += arg.type.cursor_length() if arg.type.is_cursor() else 1

        for i, arg in enumerate(output_args):
            arg.arg_pos = i
            arg.kind = "output"

        return tf_node.UdtfNode(name, input_args, output_args, annotations, templates, sizer, self.line)

    def parse_args(self):
        """fmt: off

        args: arg IDENTIFIER ("," arg)*

        fmt: on
        """
        args = []
        args.append(self.parse_arg())
        while not self.is_at_end() and self.match(Token.COMMA):
            curr = self._curr
            self.consume(Token.COMMA)
            self.parse_type()  # assuming that we are not ending with COMMA
            if not self.is_at_end() and self.match(Token.EQUAL):
                # arg type cannot be assigned, so this must be a template specification
                self._curr = curr  # step back and let the code below parse the templates
                break
            else:
                self._curr = curr + 1  # step back from self.parse_type(), parse_arg will parse it again
                args.append(self.parse_arg())
        return args

    def parse_arg(self):
        """fmt: off

        arg: type IDENTIFIER? ("|" annotation)*

        fmt: on
        """
        typ = self.parse_type()

        annotations = []

        if not self.is_at_end() and self.match(Token.IDENTIFIER):
            name = self.parse_identifier()
            annotations.append(tf_node.AnnotationNode('name', name))

        while not self.is_at_end() and self.match(Token.VBAR):
            ahead = self.lookahead()
            if ahead.type == Token.IDENTIFIER and ahead.lexeme == 'output_row_size':
                break
            self.consume(Token.VBAR)
            annotations.append(self.parse_annotation())

        return tf_node.ArgNode(typ, annotations)

    def parse_type(self):
        """fmt: off

        type: composed
            | primitive

        fmt: on
        """
        curr = self._curr  # save state
        primitive = self.parse_primitive()
        if self.is_at_end():
            return primitive

        if not self.match(Token.LESS):
            return primitive

        self._curr = curr  # return state

        return self.parse_composed()

    def parse_composed(self):
        """fmt: off

        composed: "Cursor" "<" arg ("," arg)* ">"
                | IDENTIFIER "<" type ("," type)* ">"

        fmt: on
        """
        idtn = self.parse_identifier()
        self.consume(Token.LESS)
        if is_identifier_cursor(idtn):
            inner = [self.parse_arg()]
            while self.match(Token.COMMA):
                self.consume(Token.COMMA)
                inner.append(self.parse_arg())
        else:
            inner = [self.parse_type()]
            while self.match(Token.COMMA):
                self.consume(Token.COMMA)
                inner.append(self.parse_type())
        self.consume(Token.GREATER)
        return tf_node.ComposedNode(idtn, inner)

    def parse_primitive(self):
        """fmt: off

        primitive: IDENTIFIER
                 | NUMBER
                 | STRING
                 | BOOLEAN

        fmt: on
        """
        if self.match(Token.IDENTIFIER):
            lexeme = self.parse_identifier()
        elif self.match(Token.NUMBER):
            lexeme = self.parse_number()
        elif self.match(Token.STRING):
            lexeme = self.parse_string()
        elif self.match(Token.BOOLEAN):
            lexeme = self.parse_boolean()
        else:
            raise self.raise_parser_error()
        return tf_node.PrimitiveNode(lexeme)

    def parse_templates(self):
        """fmt: off

        templates: template ("," template)*

        fmt: on
        """
        T = []
        T.append(self.parse_template())
        while not self.is_at_end() and self.match(Token.COMMA):
            self.consume(Token.COMMA)
            T.append(self.parse_template())
        return T

    def parse_template(self):
        """fmt: off

        template: IDENTIFIER "=" "[" IDENTIFIER ("," IDENTIFIER)* "]"

        fmt: on
        """
        key = self.parse_identifier()
        types = []
        self.consume(Token.EQUAL)
        self.consume(Token.LSQB)
        types.append(self.parse_identifier())
        while self.match(Token.COMMA):
            self.consume(Token.COMMA)
            types.append(self.parse_identifier())
        self.consume(Token.RSQB)
        return tf_node.TemplateNode(key, tuple(types))

    def parse_annotation(self):
        """fmt: off

        annotation: IDENTIFIER "=" IDENTIFIER ("<" NUMBER ("," NUMBER) ">")?
                  | IDENTIFIER "=" "[" PRIMITIVE? ("," PRIMITIVE)* "]"
                  | "require" "=" STRING
                  | "default" "=" STRING | NUMBER | BOOLEAN

        fmt: on
        """
        key = self.parse_identifier()
        self.consume(Token.EQUAL)

        if key == "require":
            value = self.parse_string()
        elif key == "default":
            if self.match(Token.NUMBER):
                value = self.parse_number()
            elif self.match(Token.STRING):
                value = self.parse_string()
            elif self.match(Token.BOOLEAN):
                value = self.parse_boolean()
            else:
                self.raise_parser_error(
                    'Unable to parse value in \"default\" annotation.\n'
                    'Expected type NUMBER, STRING or BOOLEAN.\n'
                    'Found token: "%s" of type "%s" \n'
                    % (self.current_token().lexeme, Token.tok_name(self.current_token().type))
                )
        elif not self.is_at_end() and self.match(Token.LSQB):
            value = []
            self.consume(Token.LSQB)
            if not self.match(Token.RSQB):
                value.append(self.parse_primitive())
                while self.match(Token.COMMA):
                    self.consume(Token.COMMA)
                    value.append(self.parse_primitive())
            self.consume(Token.RSQB)
        else:
            value = self.parse_identifier()
            if not self.is_at_end() and self.match(Token.LESS):
                self.consume(Token.LESS)
                if self.match(Token.GREATER):
                    value += "<%s>" % (-1)  # Signifies no input
                else:
                    num1 = self.parse_number()
                    if self.match(Token.COMMA):
                        self.consume(Token.COMMA)
                        num2 = self.parse_number()
                        value += "<%s,%s>" % (num1, num2)
                    else:
                        value += "<%s>" % (num1)
                self.consume(Token.GREATER)
        return tf_node.AnnotationNode(key, value)

    def parse_identifier(self):
        """ fmt: off

        IDENTIFIER: [A-Za-z_][A-Za-z0-9_]*

        fmt: on
        """
        token = self.consume(Token.IDENTIFIER)
        return token.lexeme

    def parse_string(self):
        """ fmt: off

        STRING: \".*?\"

        fmt: on
        """
        token = self.consume(Token.STRING)
        return token.lexeme

    def parse_number(self):
        """ fmt: off

        NUMBER: [-]([0-9]*[.])?[0-9]+

        fmt: on
        """
        token = self.consume(Token.NUMBER)
        return token.lexeme

    def parse_boolean(self):
        """ fmt: off

        BOOLEAN: \bTrue\b|\bFalse\b

        fmt: on
        """
        token = self.consume(Token.BOOLEAN)
        # Make sure booleans are normalized to "False" or "True" regardless
        # of original capitalization, so they can be properly parsed during
        # typechecking
        new_token = token.lexeme.lower().capitalize()
        return new_token

    def parse(self):
        """fmt: off

        udtf: IDENTIFIER "(" (args)? ")" ("|" annotation)* "->" args ("," templates)? ("|" "output_row_size" "=" primitive)?

        args: arg ("," arg)*

        arg: type IDENTIFIER? ("|" annotation)*

        type: composed
            | primitive

        composed: "Cursor" "<" arg ("," arg)* ">"
                | IDENTIFIER "<" type ("," type)* ">"

        primitive: IDENTIFIER
                 | NUMBER
                 | STRING
                 | BOOLEAN

        annotation: IDENTIFIER "=" IDENTIFIER ("<" NUMBER ("," NUMBER) ">")?
                  | IDENTIFIER "=" "[" PRIMITIVE? ("," PRIMITIVE)* "]"
                  | "require" "=" STRING
                  | "default" "=" STRING | NUMBER | BOOLEAN

        templates: template ("," template)
        template: IDENTIFIER "=" "[" IDENTIFIER ("," IDENTIFIER)* "]"

        IDENTIFIER: [A-Za-z_][A-Za-z0-9_]*
        NUMBER: [0-9]+
        STRING: \".*?\"
        BOOLEAN: \bTrue\b|\bFalse\b

        fmt: on
        """
        self._curr = 0
        udtf = self.parse_udtf()

        # set parent
        udtf.parent = None
        d = deque()
        d.append(udtf)
        while d:
            node = d.pop()
            if isinstance(node, Iterable):
                for child in node:
                    child.parent = node
                    d.append(child)
        return udtf
