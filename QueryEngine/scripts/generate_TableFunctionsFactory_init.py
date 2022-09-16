"""Given a list of input files, scan for lines containing UDTF
specification statements in the following form:

  UDTF: function_name(<arguments>) -> <output column types> (, <template type specifications>)?

where <arguments> is a comma-separated list of argument types. The
argument types specifications are:

- scalar types:
    Int8, Int16, Int32, Int64, Float, Double, Bool, TextEncodingDict, etc
- column types:
    ColumnInt8, ColumnInt16, ColumnInt32, ColumnInt64, ColumnFloat, ColumnDouble, ColumnBool, etc
- column list types:
    ColumnListInt8, ColumnListInt16, ColumnListInt32, ColumnListInt64, ColumnListFloat, ColumnListDouble, ColumnListBool, etc
- cursor type:
    Cursor<t0, t1, ...>
  where t0, t1 are column or column list types
- output buffer size parameter type:
    RowMultiplier<i>, ConstantParameter<i>, Constant<i>, TableFunctionSpecifiedParameter<i>
  where i is a literal integer.

The output column types is a comma-separated list of column types, see above.

In addition, the following equivalents are suppored:

  Column<T> == ColumnT
  ColumnList<T> == ColumnListT
  Cursor<T, V, ...> == Cursor<ColumnT, ColumnV, ...>
  int8 == int8_t == Int8, etc
  float == Float, double == Double, bool == Bool
  T == ColumnT for output column types
  RowMultiplier == RowMultiplier<i> where i is the one-based position of the sizer argument
  when no sizer argument is provided, Constant<1> is assumed

Argument types can be annotated using `|' (bar) symbol after an
argument type specification. An annotation is specified by a label and
a value separated by `=' (equal) symbol. Multiple annotations can be
specified by using `|` (bar) symbol as the annotations separator.
Supported annotation labels are:

- name: to specify argument name
- input_id: to specify the dict id mapping for output TextEncodingDict columns.

If argument type follows an identifier, it will be mapped to name
annotations. For example, the following argument type specifications
are equivalent:

  Int8 a
  Int8 | name=a

Template type specifications is a comma separated list of template
type assignments where values are lists of argument type names. For
instance:

  T = [Int8, Int16, Int32, Float], V = [Float, Double]

"""
# Author: Pearu Peterson
# Created: January 2021


import os
import sys
import itertools
import copy
from abc import abstractmethod

from collections import deque, namedtuple

if sys.version_info > (3, 0):
    from abc import ABC
    from collections.abc import Iterable
else:
    from abc import ABCMeta as ABC
    from collections import Iterable

# fmt: off
separator = '$=>$'

Signature = namedtuple('Signature', ['name', 'inputs', 'outputs', 'input_annotations', 'output_annotations', 'function_annotations', 'sizer'])

OutputBufferSizeTypes = '''
kConstant, kUserSpecifiedConstantParameter, kUserSpecifiedRowMultiplier, kTableFunctionSpecifiedParameter, kPreFlightParameter
'''.strip().replace(' ', '').split(',')

SupportedAnnotations = '''
input_id, name, fields, require, range
'''.strip().replace(' ', '').split(',')

# TODO: support `gpu`, `cpu`, `template` as function annotations
SupportedFunctionAnnotations = '''
filter_table_function_transpose, uses_manager
'''.strip().replace(' ', '').split(',')

translate_map = dict(
    Constant='kConstant',
    PreFlight='kPreFlightParameter',
    ConstantParameter='kUserSpecifiedConstantParameter',
    RowMultiplier='kUserSpecifiedRowMultiplier',
    UserSpecifiedConstantParameter='kUserSpecifiedConstantParameter',
    UserSpecifiedRowMultiplier='kUserSpecifiedRowMultiplier',
    TableFunctionSpecifiedParameter='kTableFunctionSpecifiedParameter',
    short='Int16',
    int='Int32',
    long='Int64',
)
for t in ['Int8', 'Int16', 'Int32', 'Int64', 'Float', 'Double', 'Bool',
          'TextEncodingDict', 'TextEncodingNone']:
    translate_map[t.lower()] = t
    if t.startswith('Int'):
        translate_map[t.lower() + '_t'] = t


class Declaration:
    """Holds a `TYPE | ANNOTATIONS`-like structure.
    """
    def __init__(self, type, annotations=[]):
        self.type = type
        self.annotations = annotations

    @property
    def name(self):
        return self.type.name

    @property
    def args(self):
        return self.type.args

    def format_sizer(self):
        return self.type.format_sizer()

    def __repr__(self):
        return 'Declaration(%r, ann=%r)' % (self.type, self.annotations)

    def __str__(self):
        if not self.annotations:
            return str(self.type)
        return '%s | %s' % (self.type, ' | '.join(map(str, self.annotations)))

    def tostring(self):
        return self.type.tostring()

    def apply_column(self):
        return self.__class__(self.type.apply_column(), self.annotations)

    def apply_namespace(self, ns='ExtArgumentType'):
        return self.__class__(self.type.apply_namespace(ns), self.annotations)

    def get_cpp_type(self):
        return self.type.get_cpp_type()

    def format_cpp_type(self, idx, use_generic_arg_name=False, is_input=True):
        real_arg_name = dict(self.annotations).get('name', None)
        return self.type.format_cpp_type(idx,
                                         use_generic_arg_name=use_generic_arg_name,
                                         real_arg_name=real_arg_name,
                                         is_input=is_input)

    def __getattr__(self, name):
        if name.startswith('is_'):
            return getattr(self.type, name)
        raise AttributeError(name)


def tostring(obj):
    return obj.tostring()


class Bracket:
    """Holds a `NAME<ARGS>`-like structure.
    """

    def __init__(self, name, args=None):
        assert isinstance(name, str)
        assert isinstance(args, tuple) or args is None, args
        self.name = name
        self.args = args

    def __repr__(self):
        return 'Bracket(%r, args=%r)' % (self.name, self.args)

    def __str__(self):
        if not self.args:
            return self.name
        return '%s<%s>' % (self.name, ', '.join(map(str, self.args)))

    def tostring(self):
        if not self.args:
            return self.name
        return '%s<%s>' % (self.name, ', '.join(map(tostring, self.args)))

    def normalize(self, kind='input'):
        """Normalize bracket for given kind
        """
        assert kind in ['input', 'output'], kind
        if self.is_column_any() and self.args:
            return Bracket(self.name + ''.join(map(str, self.args)))
        if kind == 'input':
            if self.name == 'Cursor':
                args = [(a if a.is_column_any() else Bracket('Column', args=(a,))).normalize(kind=kind) for a in self.args]
                return Bracket(self.name, tuple(args))
        if kind == 'output':
            if not self.is_column_any():
                return Bracket('Column', args=(self,)).normalize(kind=kind)
        return self

    def apply_cursor(self):
        """Apply cursor to a non-cursor column argument type.
        TODO: this method is currently unused but we should apply
        cursor to all input column arguments in order to distingush
        signatures like:
          foo(Cursor(Column<int32>, Column<float>)) -> Column<int32>
          foo(Cursor(Column<int32>), Cursor(Column<float>)) -> Column<int32>
        that at the moment are treated as the same :(
        """
        if self.is_column():
            return Bracket('Cursor', args=(self,))
        return self

    def apply_column(self):
        if not self.is_column() and not self.is_column_list():
            return Bracket('Column' + self.name)
        return self

    def apply_namespace(self, ns='ExtArgumentType'):
        if self.name == 'Cursor':
            return Bracket(ns + '::' + self.name, args=tuple(a.apply_namespace(ns=ns) for a in self.args))
        if not self.name.startswith(ns + '::'):
            return Bracket(ns + '::' + self.name)
        return self

    def is_cursor(self):
        return self.name.rsplit("::", 1)[-1] == 'Cursor'

    def is_array(self):
        return self.name.rsplit("::", 1)[-1].startswith('Array')

    def is_column_any(self):
        return self.name.rsplit("::", 1)[-1].startswith('Column')

    def is_column_list(self):
        return self.name.rsplit("::", 1)[-1].startswith('ColumnList')

    def is_column(self):
        return self.name.rsplit("::", 1)[-1].startswith('Column') and not self.is_column_list()

    def is_any_text_encoding_dict(self):
        return self.name.rsplit("::", 1)[-1].endswith('TextEncodingDict')

    def is_array_text_encoding_dict(self):
        return self.name.rsplit("::", 1)[-1] == 'ArrayTextEncodingDict'

    def is_column_text_encoding_dict(self):
        return self.name.rsplit("::", 1)[-1] == 'ColumnTextEncodingDict'

    def is_column_array_text_encoding_dict(self):
        return self.name.rsplit("::", 1)[-1] == 'ColumnArrayTextEncodingDict'

    def is_column_list_text_encoding_dict(self):
        return self.name.rsplit("::", 1)[-1] == 'ColumnListTextEncodingDict'

    def is_output_buffer_sizer(self):
        return self.name.rsplit("::", 1)[-1] in OutputBufferSizeTypes

    def is_row_multiplier(self):
        return self.name.rsplit("::", 1)[-1] == 'kUserSpecifiedRowMultiplier'

    def is_arg_sizer(self):
        return self.name.rsplit("::", 1)[-1] == 'kPreFlightParameter'

    def is_user_specified(self):
        # Return True if given argument cannot specified by user
        if self.is_output_buffer_sizer():
            return self.name.rsplit("::", 1)[-1] not in ('kConstant', 'kTableFunctionSpecifiedParameter', 'kPreFlightParameter')
        return True

    def format_sizer(self):
        val = 0 if self.is_arg_sizer() else self.args[0]
        return 'TableFunctionOutputRowSizer{OutputBufferSizeType::%s, %s}' % (self.name, val)

    def get_cpp_type(self):
        name = self.name.rsplit("::", 1)[-1]
        clsname = None
        subclsname = None
        if name.startswith('ColumnList'):
            name = name.lstrip('ColumnList')
            clsname = 'ColumnList'
        elif name.startswith('Column'):
            name = name.lstrip('Column')
            clsname = 'Column'
        if name.startswith('Array'):
            name = name.lstrip('Array')
            if clsname is None:
                clsname = 'Array'
            else:
                subclsname = 'Array'

        if name.startswith('Bool'):
            ctype = name.lower()
        elif name.startswith('Int'):
            ctype = name.lower() + '_t'
        elif name in ['Double', 'Float']:
            ctype = name.lower()
        elif name == 'TextEncodingDict':
            ctype = name
        elif name == 'TextEncodingNone':
            ctype = name
        elif name == 'Timestamp':
            ctype = name
        elif name == 'DayTimeInterval':
            ctype = name
        elif name == 'YearMonthTimeInterval':
            ctype = name
        else:
            raise NotImplementedError(self)
        if clsname is None:
            return ctype
        if subclsname is None:
            return '%s<%s>' % (clsname, ctype)
        return '%s<%s<%s>>' % (clsname, subclsname, ctype)

    def format_cpp_type(self, idx, use_generic_arg_name=False, real_arg_name=None, is_input=True):
        col_typs = ('Column', 'ColumnList')
        literal_ref_typs = ('TextEncodingNone',)
        if use_generic_arg_name:
            arg_name = 'input' + str(idx) if is_input else 'output' + str(idx)
        elif real_arg_name is not None:
            arg_name = real_arg_name
        else:
            # in some cases, the real arg name is not specified
            arg_name = 'input' + str(idx) if is_input else 'output' + str(idx)
        const = 'const ' if is_input else ''
        cpp_type = self.get_cpp_type()
        if any(cpp_type.startswith(t) for t in col_typs + literal_ref_typs):
            return '%s%s& %s' % (const, cpp_type, arg_name), arg_name
        else:
            return '%s %s' % (cpp_type, arg_name), arg_name

    @classmethod
    def parse(cls, typ):
        """typ is a string in format NAME<ARGS> or NAME

        Returns Bracket instance.
        """
        i = typ.find('<')
        if i == -1:
            name = typ.strip()
            args = None
        else:
            assert typ.endswith('>'), typ
            name = typ[:i].strip()
            args = []
            rest = typ[i + 1:-1].strip()
            while rest:
                i = find_comma(rest)
                if i == -1:
                    a, rest = rest, ''
                else:
                    a, rest = rest[:i].rstrip(), rest[i + 1:].lstrip()
                args.append(cls.parse(a))
            args = tuple(args)

        name = translate_map.get(name, name)
        return cls(name, args)


def find_comma(line):
    d = 0
    for i, c in enumerate(line):
        if c in '<([{':
            d += 1
        elif c in '>)]{':
            d -= 1
        elif d == 0 and c == ',':
            return i
    return -1


def line_is_incomplete(line):
    # TODO: try to parse the line to be certain about completeness.
    # `$=>$' is used to separate the UDTF signature and the expected result
    return line.endswith(',') or line.endswith('->') or line.endswith(separator) or line.endswith('|')


def is_identifier_cursor(identifier):
    return identifier.lower() == 'cursor'


# fmt: on


class TokenizeException(Exception):
    pass


class ParserException(Exception):
    pass


class TransformerException(Exception):
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
            elif self.is_digit():
                self.consume_number()
            elif self.is_token_string():
                self.consume_string()
            elif self.is_token_identifier():
                self.consume_identifier()
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
        NUMBER: [0-9]+
        """
        while True:
            char = self.lookahead()
            if char and char.isdigit():
                self.advance()
            else:
                break
        self.add_token(Token.NUMBER)
        self.advance()

    def consume_identifier(self):
        """
        IDENTIFIER: [A-Za-z_][A-Za-z0-9_]*
        """
        while True:
            char = self.lookahead()
            if char and char.isalnum() or char == "_":
                self.advance()
            else:
                break
        self.add_token(Token.IDENTIFIER)
        self.advance()

    def is_token_identifier(self):
        return self.peek().isalpha() or self.peek() == "_"

    def is_token_string(self):
        return self.peek() == '"'

    def is_digit(self):
        return self.peek().isdigit()

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


class AstVisitor(object):
    __metaclass__ = ABC

    @abstractmethod
    def visit_udtf_node(self, node):
        pass

    @abstractmethod
    def visit_composed_node(self, node):
        pass

    @abstractmethod
    def visit_arg_node(self, node):
        pass

    @abstractmethod
    def visit_primitive_node(self, node):
        pass

    @abstractmethod
    def visit_annotation_node(self, node):
        pass

    @abstractmethod
    def visit_template_node(self, node):
        pass


class AstTransformer(AstVisitor):
    """Only overload the methods you need"""

    def visit_udtf_node(self, udtf_node):
        udtf = copy.copy(udtf_node)
        udtf.inputs = [arg.accept(self) for arg in udtf.inputs]
        udtf.outputs = [arg.accept(self) for arg in udtf.outputs]
        if udtf.templates:
            udtf.templates = [t.accept(self) for t in udtf.templates]
        udtf.annotations = [annot.accept(self) for annot in udtf.annotations]
        return udtf

    def visit_composed_node(self, composed_node):
        c = copy.copy(composed_node)
        c.inner = [i.accept(self) for i in c.inner]
        return c

    def visit_arg_node(self, arg_node):
        arg_node = copy.copy(arg_node)
        arg_node.type = arg_node.type.accept(self)
        if arg_node.annotations:
            arg_node.annotations = [a.accept(self) for a in arg_node.annotations]
        return arg_node

    def visit_primitive_node(self, primitive_node):
        return copy.copy(primitive_node)

    def visit_template_node(self, template_node):
        return copy.copy(template_node)

    def visit_annotation_node(self, annotation_node):
        return copy.copy(annotation_node)


class AstPrinter(AstVisitor):
    """Returns a line formatted. Useful for testing"""

    def visit_udtf_node(self, udtf_node):
        name = udtf_node.name
        inputs = ", ".join([arg.accept(self) for arg in udtf_node.inputs])
        outputs = ", ".join([arg.accept(self) for arg in udtf_node.outputs])
        annotations = "| ".join([annot.accept(self) for annot in udtf_node.annotations])
        sizer = " | " + udtf_node.sizer.accept(self) if udtf_node.sizer else ""
        if annotations:
            annotations = ' | ' + annotations
        if udtf_node.templates:
            templates = ", ".join([t.accept(self) for t in udtf_node.templates])
            return "%s(%s)%s -> %s, %s%s" % (name, inputs, annotations, outputs, templates, sizer)
        else:
            return "%s(%s)%s -> %s%s" % (name, inputs, annotations, outputs, sizer)

    def visit_template_node(self, template_node):
        # T=[T1, T2, ..., TN]
        key = template_node.key
        types = ['"%s"' % typ for typ in template_node.types]
        return "%s=[%s]" % (key, ", ".join(types))

    def visit_annotation_node(self, annotation_node):
        # key=value
        key = annotation_node.key
        value = annotation_node.value
        if isinstance(value, list):
            return "%s=[%s]" % (key, ','.join([v.accept(self) for v in value]))
        return "%s=%s" % (key, value)

    def visit_arg_node(self, arg_node):
        # type | annotation
        typ = arg_node.type.accept(self)
        if arg_node.annotations:
            ann = " | ".join([a.accept(self) for a in arg_node.annotations])
            s = "%s | %s" % (typ, ann)
        else:
            s = "%s" % (typ,)
        # insert input_id=args<0> if input_id is not specified
        if s == "ColumnTextEncodingDict" and arg_node.kind == "output":
            return s + " | input_id=args<0>"
        return s

    def visit_composed_node(self, composed_node):
        T = composed_node.inner[0].accept(self)
        if composed_node.is_array():
            # Array<T>
            assert len(composed_node.inner) == 1
            return "Array" + T
        if composed_node.is_column():
            # Column<T>
            assert len(composed_node.inner) == 1
            return "Column" + T
        if composed_node.is_column_list():
            # ColumnList<T>
            assert len(composed_node.inner) == 1
            return "ColumnList" + T
        if composed_node.is_output_buffer_sizer():
            # kConstant<N>
            N = T
            assert len(composed_node.inner) == 1
            return translate_map.get(composed_node.type) + "<%s>" % (N,)
        if composed_node.is_cursor():
            # Cursor<T1, T2, ..., TN>
            Ts = ", ".join([i.accept(self) for i in composed_node.inner])
            return "Cursor<%s>" % (Ts)
        raise ValueError(composed_node)

    def visit_primitive_node(self, primitive_node):
        t = primitive_node.type
        if primitive_node.is_output_buffer_sizer():
            # arg_pos is zero-based
            return translate_map.get(t, t) + "<%d>" % (
                primitive_node.get_parent(ArgNode).arg_pos + 1,
            )
        return translate_map.get(t, t)


class AstDebugger(AstTransformer):
    """Like AstPrinter but returns a node instead of a string
    """


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


class TemplateTransformer(AstTransformer):
    """Expand template definition into multiple inputs"""

    def visit_udtf_node(self, udtf_node):
        if not udtf_node.templates:
            return udtf_node

        udtfs = dict()

        d = dict([(node.key, node.types) for node in udtf_node.templates])
        name = udtf_node.name

        for product in product_dict(**d):
            self.mapping_dict = product
            inputs = [input_arg.accept(self) for input_arg in udtf_node.inputs]
            outputs = [output_arg.accept(self) for output_arg in udtf_node.outputs]
            udtf = UdtfNode(name, inputs, outputs, udtf_node.annotations, None, udtf_node.sizer, udtf_node.line)
            udtfs[str(udtf)] = udtf
            self.mapping_dict = {}

        udtfs = list(udtfs.values())

        if len(udtfs) == 1:
            return udtfs[0]

        return udtfs

    def visit_composed_node(self, composed_node):
        typ = composed_node.type
        typ = self.mapping_dict.get(typ, typ)

        inner = [i.accept(self) for i in composed_node.inner]
        return composed_node.copy(typ, inner)

    def visit_primitive_node(self, primitive_node):
        typ = primitive_node.type
        typ = self.mapping_dict.get(typ, typ)
        return primitive_node.copy(typ)


class FixRowMultiplierPosArgTransformer(AstTransformer):
    def visit_primitive_node(self, primitive_node):
        """
        * Fix kUserSpecifiedRowMultiplier without a pos arg
        """
        t = primitive_node.type

        if primitive_node.is_output_buffer_sizer():
            pos = PrimitiveNode(str(primitive_node.get_parent(ArgNode).arg_pos + 1))
            node = ComposedNode(t, inner=[pos])
            return node

        return primitive_node


class RenameNodesTransformer(AstTransformer):
    def visit_primitive_node(self, primitive_node):
        """
        * Rename nodes using translate_map as dictionary
            int -> Int32
            float -> Float
        """
        t = primitive_node.type
        return primitive_node.copy(translate_map.get(t, t))


class TextEncodingDictTransformer(AstTransformer):
    def visit_udtf_node(self, udtf_node):
        """
        * Add default_input_id to Column(List)<TextEncodingDict> without one
        """
        udtf_node = super(type(self), self).visit_udtf_node(udtf_node)
        # add default input_id
        default_input_id = None
        for idx, t in enumerate(udtf_node.inputs):

            if not isinstance(t.type, ComposedNode):
                continue
            if default_input_id is not None:
                pass
            elif t.type.is_column_text_encoding_dict() or t.type.is_column_array_text_encoding_dict():
                default_input_id = AnnotationNode('input_id', 'args<%s>' % (idx,))
            elif t.type.is_column_list_text_encoding_dict():
                default_input_id = AnnotationNode('input_id', 'args<%s, 0>' % (idx,))

        for t in udtf_node.outputs:
            if isinstance(t.type, ComposedNode) and t.type.is_any_text_encoding_dict():
                for a in t.annotations:
                    if a.key == 'input_id':
                        break
                else:
                    if default_input_id is None:
                        raise TypeError('Cannot parse line "%s".\n'
                                        'Missing TextEncodingDict input?' %
                                        (udtf_node.line))
                    t.annotations.append(default_input_id)

        return udtf_node


class FieldAnnotationTransformer(AstTransformer):

    def visit_udtf_node(self, udtf_node):
        """
        * Generate fields annotation to Cursor if non-existing
        """
        udtf_node = super(type(self), self).visit_udtf_node(udtf_node)

        for t in udtf_node.inputs:

            if not isinstance(t.type, ComposedNode):
                continue

            if t.type.is_cursor() and t.get_annotation('fields') is None:
                fields = list(PrimitiveNode(a.get_annotation('name', 'field%s' % i)) for i, a in enumerate(t.type.inner))
                t.annotations.append(AnnotationNode('fields', fields))

        return udtf_node
        
class SupportedAnnotationsTransformer(AstTransformer):
    """
    * Checks for supported annotations in a UDTF
    """
    def visit_udtf_node(self, udtf_node):
        for t in udtf_node.inputs:
            for a in t.annotations:
                if a.key not in SupportedAnnotations:
                    raise TransformerException('unknown input annotation: `%s`' % (a.key))
        for t in udtf_node.outputs:
            for a in t.annotations:
                if a.key not in SupportedAnnotations:
                    raise TransformerException('unknown output annotation: `%s`' % (a.key))
        for annot in udtf_node.annotations:
            if annot.key not in SupportedFunctionAnnotations:
                raise TransformerException('unknown function annotation: `%s`' % (annot.key))
            if annot.value.lower() in ['enable', 'on', '1', 'true']:
                annot.value = '1'
            elif annot.value.lower() in ['disable', 'off', '0', 'false']:
                annot.value = '0'
        return udtf_node


class RangeAnnotationTransformer(AstTransformer):
    """
    * Append require annotation if range is used
    """
    def visit_arg_node(self, arg_node):
        for ann in arg_node.annotations:
            if ann.key == 'range':
                name = arg_node.get_annotation('name')
                if name is None:
                    raise TransformerException('"range" requires a named argument')

                l = ann.value
                if len(l) == 2:
                    lo, hi = ann.value
                    value = '"{lo} <= {name} && {name} <= {hi}"'.format(lo=lo, hi=hi, name=name)
                else:
                    raise TransformerException('"range" requires an interval. Got {l}'.format(l=l))
                arg_node.set_annotation('require', value)
        return arg_node


class DeclBracketTransformer(AstTransformer):

    def visit_udtf_node(self, udtf_node):
        name = udtf_node.name
        inputs = []
        input_annotations = []
        outputs = []
        output_annotations = []
        function_annotations = []
        sizer = udtf_node.sizer

        for i in udtf_node.inputs:
            decl = i.accept(self)
            inputs.append(decl)
            input_annotations.append(decl.annotations)

        for o in udtf_node.outputs:
            decl = o.accept(self)
            outputs.append(decl.type)
            output_annotations.append(decl.annotations)

        for annot in udtf_node.annotations:
            annot = annot.accept(self)
            function_annotations.append(annot)

        return Signature(name, inputs, outputs, input_annotations, output_annotations, function_annotations, sizer)

    def visit_arg_node(self, arg_node):
        t = arg_node.type.accept(self)
        anns = [a.accept(self) for a in arg_node.annotations]
        return Declaration(t, anns)

    def visit_composed_node(self, composed_node):
        typ = translate_map.get(composed_node.type, composed_node.type)
        inner = [i.accept(self) for i in composed_node.inner]
        if composed_node.is_cursor():
            inner = list(map(lambda x: x.apply_column(), inner))
            return Bracket(typ, args=tuple(inner))
        elif composed_node.is_output_buffer_sizer():
            return Bracket(typ, args=tuple(inner))
        else:
            return Bracket(typ + str(inner[0]))

    def visit_primitive_node(self, primitive_node):
        t = primitive_node.type
        return Bracket(t)

    def visit_annotation_node(self, annotation_node):
        key = annotation_node.key
        value = annotation_node.value
        return (key, value)


class Node(object):

    __metaclass__ = ABC

    @abstractmethod
    def accept(self, visitor):
        pass

    @abstractmethod
    def __str__(self):
        pass

    def get_parent(self, cls):
        if isinstance(self, cls):
            return self

        if self.parent is not None:
            return self.parent.get_parent(cls)

        raise ValueError("could not find parent with given class %s" % (cls))

    def copy(self, *args):
        other = self.__class__(*args)

        # copy parent and arg_pos
        for attr in ['parent', 'arg_pos']:
            if attr in self.__dict__:
                setattr(other, attr, getattr(self, attr))

        return other


class IterableNode(Iterable):
    pass


class UdtfNode(Node, IterableNode):

    def __init__(self, name, inputs, outputs, annotations, templates, sizer, line):
        """
        Parameters
        ----------
        name : str
        inputs : list[ArgNode]
        outputs : list[ArgNode]
        annotations : Optional[List[AnnotationNode]]
        templates : Optional[list[TemplateNode]]
        sizer : Optional[str]
        line: str
        """
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.annotations = annotations
        self.templates = templates
        self.sizer = sizer
        self.line = line

    def accept(self, visitor):
        return visitor.visit_udtf_node(self)

    def __str__(self):
        name = self.name
        inputs = [str(i) for i in self.inputs]
        outputs = [str(o) for o in self.outputs]
        annotations = [str(a) for a in self.annotations]
        sizer = "| %s" % str(self.sizer) if self.sizer else ""
        if self.templates:
            templates = [str(t) for t in self.templates]
            if annotations:
                return "UDTF: %s (%s) | %s -> %s, %s %s" % (name, inputs, annotations, outputs, templates, sizer)
            else:
                return "UDTF: %s (%s) -> %s, %s %s" % (name, inputs, outputs, templates, sizer)
        else:
            if annotations:
                return "UDTF: %s (%s) | %s -> %s %s" % (name, inputs, annotations, outputs, sizer)
            else:
                return "UDTF: %s (%s) -> %s %s" % (name, inputs, outputs, sizer)

    def __iter__(self):
        for i in self.inputs:
            yield i
        for o in self.outputs:
            yield o
        for a in self.annotations:
            yield a
        if self.templates:
            for t in self.templates:
                yield t

    __repr__ = __str__


class ArgNode(Node, IterableNode):

    def __init__(self, type, annotations):
        """
        Parameters
        ----------
        type : TypeNode
        annotations : List[AnnotationNode]
        """
        self.type = type
        self.annotations = annotations
        self.arg_pos = None

    def accept(self, visitor):
        return visitor.visit_arg_node(self)

    def __str__(self):
        t = str(self.type)
        anns = ""
        if self.annotations:
            anns = " | ".join([str(a) for a in self.annotations])
            return "ArgNode(%s | %s)" % (t, anns)
        return "ArgNode(%s)" % (t)

    def __iter__(self):
        yield self.type
        for a in self.annotations:
            yield a

    __repr__ = __str__

    def get_annotation(self, key, default=None):
        for a in self.annotations:
            if a.key == key:
                return a.value
        return default

    def set_annotation(self, key, value):
        found = False
        for i, a in enumerate(self.annotations):
            if a.key == key:
                assert not found, (i, a)  # annotations with the same key not supported
                self.annotations[i] = AnnotationNode(key, value)
                found = True
        if not found:
            self.annotations.append(AnnotationNode(key, value))


class TypeNode(Node):
    def is_array(self):
        return self.type == "Array"

    def is_column_any(self):
        return self.is_column() or self.is_column_list()

    def is_column(self):
        return self.type == "Column"

    def is_column_list(self):
        return self.type == "ColumnList"

    def is_cursor(self):
        return self.type == "Cursor"

    def is_output_buffer_sizer(self):
        t = self.type
        return translate_map.get(t, t) in OutputBufferSizeTypes


class PrimitiveNode(TypeNode):

    def __init__(self, type):
        """
        Parameters
        ----------
        type : str
        """
        self.type = type

    def accept(self, visitor):
        return visitor.visit_primitive_node(self)

    def __str__(self):
        return self.accept(AstPrinter())

    def is_text_encoding_dict(self):
        return self.type == 'TextEncodingDict'

    def is_array_text_encoding_dict(self):
        return self.type == 'ArrayTextEncodingDict'

    __repr__ = __str__


class ComposedNode(TypeNode, IterableNode):

    def __init__(self, type, inner):
        """
        Parameters
        ----------
        type : str
        inner : list[TypeNode]
        """
        self.type = type
        self.inner = inner

    def accept(self, visitor):
        return visitor.visit_composed_node(self)

    def cursor_length(self):
        assert self.is_cursor()
        return len(self.inner)

    def __str__(self):
        i = ", ".join([str(i) for i in self.inner])
        return "Composed(%s<%s>)" % (self.type, i)

    def __iter__(self):
        for i in self.inner:
            yield i

    def is_text_encoding_dict(self):
        return False

    def is_array_text_encoding_dict(self):
        return self.is_array() and self.inner[0].is_text_encoding_dict()

    def is_column_text_encoding_dict(self):
        return self.is_column() and self.inner[0].is_text_encoding_dict()

    def is_column_list_text_encoding_dict(self):
        return self.is_column_list() and self.inner[0].is_text_encoding_dict()

    def is_column_array_text_encoding_dict(self):
        return self.is_column() and self.inner[0].is_array_text_encoding_dict()

    def is_any_text_encoding_dict(self):
        return self.inner[0].is_text_encoding_dict() or self.inner[0].is_array_text_encoding_dict()

    __repr__ = __str__


class AnnotationNode(Node):

    def __init__(self, key, value):
        """
        Parameters
        ----------
        key : str
        value : {str, list}
        """
        self.key = key
        self.value = value

    def accept(self, visitor):
        return visitor.visit_annotation_node(self)

    def __str__(self):
        printer = AstPrinter()
        return self.accept(printer)

    __repr__ = __str__


class TemplateNode(Node):

    def __init__(self, key, types):
        """
        Parameters
        ----------
        key : str
        types : tuple[str]
        """
        self.key = key
        self.types = types

    def accept(self, visitor):
        return visitor.visit_template_node(self)

    def __str__(self):
        printer = AstPrinter()
        return self.accept(printer)

    __repr__ = __str__


class Pipeline(object):
    def __init__(self, *passes):
        self.passes = passes

    def __call__(self, ast_list):
        if not isinstance(ast_list, list):
            ast_list = [ast_list]

        for c in self.passes:
            ast_list = [ast.accept(c()) for ast in ast_list]
            ast_list = itertools.chain.from_iterable(  # flatten the list
                map(lambda x: x if isinstance(x, list) else [x], ast_list))

        return list(ast_list)


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
            sizer = AnnotationNode(key, value=node.type)

        # set arg_pos
        i = 0
        for arg in input_args:
            arg.arg_pos = i
            arg.kind = "input"
            i += arg.type.cursor_length() if arg.type.is_cursor() else 1

        for i, arg in enumerate(output_args):
            arg.arg_pos = i
            arg.kind = "output"

        return UdtfNode(name, input_args, output_args, annotations, templates, sizer, self.line)

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
            annotations.append(AnnotationNode('name', name))

        while not self.is_at_end() and self.match(Token.VBAR):
            ahead = self.lookahead()
            if ahead.type == Token.IDENTIFIER and ahead.lexeme == 'output_row_size':
                break
            self.consume(Token.VBAR)
            annotations.append(self.parse_annotation())

        return ArgNode(typ, annotations)

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
        return ComposedNode(idtn, inner)

    def parse_primitive(self):
        """fmt: off

        primitive: IDENTIFIER
                 | NUMBER
                 | STRING

        fmt: on
        """
        if self.match(Token.IDENTIFIER):
            lexeme = self.parse_identifier()
        elif self.match(Token.NUMBER):
            lexeme = self.parse_number()
        elif self.match(Token.STRING):
            lexeme = self.parse_string()
        else:
            raise self.raise_parser_error()
        return PrimitiveNode(lexeme)

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
        return TemplateNode(key, tuple(types))

    def parse_annotation(self):
        """fmt: off

        annotation: IDENTIFIER "=" IDENTIFIER ("<" NUMBER ("," NUMBER) ">")?
                  | IDENTIFIER "=" "[" PRIMITIVE? ("," PRIMITIVE)* "]"
                  | "require" "=" STRING

        fmt: on
        """
        key = self.parse_identifier()
        self.consume(Token.EQUAL)

        if key == "require":
            value = self.parse_string()
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
                    value += "<%s>" % (-1) # Signifies no input
                else:
                    num1 = self.parse_number()
                    if self.match(Token.COMMA):
                        self.consume(Token.COMMA)
                        num2 = self.parse_number()
                        value += "<%s,%s>" % (num1, num2)
                    else:
                        value += "<%s>" % (num1)
                self.consume(Token.GREATER)
        return AnnotationNode(key, value)

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

        NUMBER: [0-9]+

        fmt: on
        """
        token = self.consume(Token.NUMBER)
        return token.lexeme

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

        annotation: IDENTIFIER "=" IDENTIFIER ("<" NUMBER ("," NUMBER) ">")?
                  | IDENTIFIER "=" "[" PRIMITIVE? ("," PRIMITIVE)* "]"
                  | "require" "=" STRING

        templates: template ("," template)
        template: IDENTIFIER "=" "[" IDENTIFIER ("," IDENTIFIER)* "]"

        IDENTIFIER: [A-Za-z_][A-Za-z0-9_]*
        NUMBER: [0-9]+
        STRING: \".*?\"

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


# fmt: off
def find_signatures(input_file):
    """Returns a list of parsed UDTF signatures."""
    signatures = []

    last_line = None
    for line in open(input_file).readlines():
        line = line.strip()
        if last_line is not None:
            line = last_line + ' ' + line
            last_line = None
        if not line.startswith('UDTF:'):
            continue
        if line_is_incomplete(line):
            last_line = line
            continue
        last_line = None
        line = line[5:].lstrip()
        i = line.find('(')
        j = line.find(')')
        if i == -1 or j == -1:
            sys.stderr.write('Invalid UDTF specification: `%s`. Skipping.\n' % (line))
            continue

        expected_result = None
        if separator in line:
            line, expected_result = line.split(separator, 1)
            expected_result = expected_result.strip().split(separator)
            expected_result = list(map(lambda s: s.strip(), expected_result))

        ast = Parser(line).parse()

        if expected_result is not None:
            # Template transformer expands templates into multiple lines
            try:
                result = Pipeline(TemplateTransformer,
                                  FieldAnnotationTransformer,
                                  TextEncodingDictTransformer,
                                  SupportedAnnotationsTransformer,
                                  RangeAnnotationTransformer,
                                  FixRowMultiplierPosArgTransformer,
                                  RenameNodesTransformer,
                                  AstPrinter)(ast)
            except TransformerException as msg:
                result = ['%s: %s' % (type(msg).__name__, msg)]
            assert len(result) == len(expected_result), "\n\tresult:   %s \n!= \n\texpected: %s" % (
                '\n\t\t  '.join(result),
                '\n\t\t  '.join(expected_result)
            )
            assert set(result) == set(expected_result), "\n\tresult:   %s != \n\texpected: %s" % (
                '\n\t\t  '.join(result),
                '\n\t\t  '.join(expected_result),
            )
            
        else:
            signature = Pipeline(TemplateTransformer,
                             FieldAnnotationTransformer,
                             TextEncodingDictTransformer,
                             SupportedAnnotationsTransformer,
                             RangeAnnotationTransformer,
                             FixRowMultiplierPosArgTransformer,
                             RenameNodesTransformer,
                             DeclBracketTransformer)(ast)

            signatures.extend(signature)

    return signatures


def format_function_args(input_types, output_types, uses_manager, use_generic_arg_name, emit_output_args):
    cpp_args = []
    name_args = []

    if uses_manager:
        cpp_args.append('TableFunctionManager& mgr')
        name_args.append('mgr')

    for idx, typ in enumerate(input_types):
        cpp_arg, name = typ.format_cpp_type(idx,
                                            use_generic_arg_name=use_generic_arg_name,
                                            is_input=True)
        cpp_args.append(cpp_arg)
        name_args.append(name)

    if emit_output_args:
        for idx, typ in enumerate(output_types):
            cpp_arg, name = typ.format_cpp_type(idx,
                                                use_generic_arg_name=use_generic_arg_name,
                                                is_input=False)
            cpp_args.append(cpp_arg)
            name_args.append(name)

    cpp_args = ', '.join(cpp_args)
    name_args = ', '.join(name_args)
    return cpp_args, name_args


def build_template_function_call(caller, called, input_types, output_types, uses_manager):
    cpp_args, name_args = format_function_args(input_types,
                                               output_types,
                                               uses_manager,
                                               use_generic_arg_name=True,
                                               emit_output_args=True)

    template = ("EXTENSION_NOINLINE int32_t\n"
                "%s(%s) {\n"
                "    return %s(%s);\n"
                "}\n") % (caller, cpp_args, called, name_args)
    return template


def build_preflight_function(fn_name, sizer, input_types, output_types, uses_manager):

    def format_error_msg(err_msg, uses_manager):
        if uses_manager:
            return "    return mgr.error_message(%s);\n" % (err_msg,)
        else:
            return "    return table_function_error(%s);\n" % (err_msg,)

    cpp_args, _ = format_function_args(input_types,
                                       output_types,
                                       uses_manager,
                                       use_generic_arg_name=False,
                                       emit_output_args=False)

    if uses_manager:
        fn = "EXTENSION_NOINLINE int32_t\n"
        fn += "%s(%s) {\n" % (fn_name.lower() + "__preflight", cpp_args)
    else:
        fn = "EXTENSION_NOINLINE int32_t\n"
        fn += "%s(%s) {\n" % (fn_name.lower() + "__preflight", cpp_args)

    for typ in input_types:
        if isinstance(typ, Declaration):
            ann = typ.annotations
            for key, value in ann:
                if key == 'require':
                    err_msg = '"Constraint `%s` is not satisfied."' % (value[1:-1])

                    fn += "  if (!(%s)) {\n" % (value[1:-1].replace('\\', ''),)
                    fn += format_error_msg(err_msg, uses_manager)
                    fn += "  }\n"

    if sizer.is_arg_sizer():
        precomputed_nrows = str(sizer.args[0])
        if '"' in precomputed_nrows:
            precomputed_nrows = precomputed_nrows[1:-1]
        # check to see if the precomputed number of rows > 0
        err_msg = '"Output size expression `%s` evaluated in a negative value."' % (precomputed_nrows)
        fn += "  auto _output_size = %s;\n" % (precomputed_nrows)
        fn += "  if (_output_size < 0) {\n"
        fn += format_error_msg(err_msg, uses_manager)
        fn += "  }\n"
        fn += "  return _output_size;\n"
    else:
        fn += "  return 0;\n"
    fn += "}\n\n"

    return fn


def must_emit_preflight_function(sig, sizer):
    if sizer.is_arg_sizer():
        return True
    for arg_annotations in sig.input_annotations:
        d = dict(arg_annotations)
        if 'require' in d.keys():
            return True
    return False


def format_annotations(annotations_):
    def fmt(k, v):
        # type(v) is not always 'str'
        if k == 'require':
            return v[1:-1]
        return v

    s = "std::vector<std::map<std::string, std::string>>{"
    s += ', '.join(('{' + ', '.join('{"%s", "%s"}' % (k, fmt(k, v)) for k, v in a) + '}') for a in annotations_)
    s += "}"
    return s


def is_template_function(sig):
    i = sig.name.rfind('_template')
    return i >= 0 and '__' in sig.name[:i + 1]


def uses_manager(sig):
    return sig.inputs and sig.inputs[0].name == 'TableFunctionManager'


def is_cpu_function(sig):
    # Any function that does not have _gpu_ suffix is a cpu function.
    i = sig.name.rfind('_gpu_')
    if i >= 0 and '__' in sig.name[:i + 1]:
        if uses_manager(sig):
            raise ValueError('Table function {} with gpu execution target cannot have TableFunctionManager argument'.format(sig.name))
        return False
    return True


def is_gpu_function(sig):
    # A function with TableFunctionManager argument is a cpu-only function
    if uses_manager(sig):
        return False
    # Any function that does not have _cpu_ suffix is a gpu function.
    i = sig.name.rfind('_cpu_')
    return not (i >= 0 and '__' in sig.name[:i + 1])


def parse_annotations(input_files):

    counter = 0

    add_stmts = []
    cpu_template_functions = []
    gpu_template_functions = []
    cpu_function_address_expressions = []
    gpu_function_address_expressions = []
    cond_fns = []

    for input_file in input_files:
        for sig in find_signatures(input_file):

            # Compute sql_types, input_types, and sizer
            sql_types_ = []
            input_types_ = []
            input_annotations = []

            sizer = None
            if sig.sizer is not None:
                expr = sig.sizer.value
                sizer = Bracket('kPreFlightParameter', (expr,))

            uses_manager = False
            for i, (t, annot) in enumerate(zip(sig.inputs, sig.input_annotations)):
                if t.is_output_buffer_sizer():
                    if t.is_user_specified():
                        sql_types_.append(Bracket.parse('int32').normalize(kind='input'))
                        input_types_.append(sql_types_[-1])
                        input_annotations.append(annot)
                    assert sizer is None  # exactly one sizer argument is allowed
                    assert len(t.args) == 1, t
                    sizer = t
                elif t.name == 'Cursor':
                    for t_ in t.args:
                        input_types_.append(t_)
                    input_annotations.append(annot)
                    sql_types_.append(Bracket('Cursor', args=()))
                elif t.name == 'TableFunctionManager':
                    if i != 0:
                        raise ValueError('{} must appear as a first argument of {}, but found it at position {}.'.format(t, sig.name, i))
                    uses_manager = True
                else:
                    input_types_.append(t)
                    input_annotations.append(annot)
                    if t.is_column_any():
                        # XXX: let Bracket handle mapping of column to cursor(column)
                        sql_types_.append(Bracket('Cursor', args=()))
                    else:
                        sql_types_.append(t)

            if sizer is None:
                name = 'kTableFunctionSpecifiedParameter'
                idx = 1  # this sizer is not actually materialized in the UDTF
                sizer = Bracket(name, (idx,))

            assert sizer is not None
            ns_output_types = tuple([a.apply_namespace(ns='ExtArgumentType') for a in sig.outputs])
            ns_input_types = tuple([t.apply_namespace(ns='ExtArgumentType') for t in input_types_])
            ns_sql_types = tuple([t.apply_namespace(ns='ExtArgumentType') for t in sql_types_])

            sig.function_annotations.append(('uses_manager', str(uses_manager).lower()))

            input_types = 'std::vector<ExtArgumentType>{%s}' % (', '.join(map(tostring, ns_input_types)))
            output_types = 'std::vector<ExtArgumentType>{%s}' % (', '.join(map(tostring, ns_output_types)))
            sql_types = 'std::vector<ExtArgumentType>{%s}' % (', '.join(map(tostring, ns_sql_types)))
            annotations = format_annotations(input_annotations + sig.output_annotations + [sig.function_annotations])

            # Notice that input_types and sig.input_types, (and
            # similarly, input_annotations and sig.input_annotations)
            # have different lengths when the sizer argument is
            # Constant or TableFunctionSpecifiedParameter. That is,
            # input_types contains all the user-specified arguments
            # while sig.input_types contains all arguments of the
            # implementation of an UDTF.

            if must_emit_preflight_function(sig, sizer):
                fn_name = '%s_%s' % (sig.name, str(counter)) if is_template_function(sig) else sig.name
                check_fn = build_preflight_function(fn_name, sizer, input_types_, sig.outputs, uses_manager)
                cond_fns.append(check_fn)

            if is_template_function(sig):
                name = sig.name + '_' + str(counter)
                counter += 1
                t = build_template_function_call(name, sig.name, input_types_, sig.outputs, uses_manager)
                address_expression = ('avoid_opt_address(reinterpret_cast<void*>(%s))' % name)
                if is_cpu_function(sig):
                    cpu_template_functions.append(t)
                    cpu_function_address_expressions.append(address_expression)
                if is_gpu_function(sig):
                    gpu_template_functions.append(t)
                    gpu_function_address_expressions.append(address_expression)
                add = ('TableFunctionsFactory::add("%s", %s, %s, %s, %s, %s, /*is_runtime:*/false);'
                       % (name, sizer.format_sizer(), input_types, output_types, sql_types, annotations))
                add_stmts.append(add)

            else:
                add = ('TableFunctionsFactory::add("%s", %s, %s, %s, %s, %s, /*is_runtime:*/false);'
                       % (sig.name, sizer.format_sizer(), input_types, output_types, sql_types, annotations))
                add_stmts.append(add)
                address_expression = ('avoid_opt_address(reinterpret_cast<void*>(%s))' % sig.name)

                if is_cpu_function(sig):
                    cpu_function_address_expressions.append(address_expression)
                if is_gpu_function(sig):
                    gpu_function_address_expressions.append(address_expression)

    return add_stmts, cpu_template_functions, gpu_template_functions, cpu_function_address_expressions, gpu_function_address_expressions, cond_fns


if len(sys.argv) < 3:

    input_files = [os.path.join(os.path.dirname(__file__), 'test_udtf_signatures.hpp')]
    print('Running tests from %s' % (', '.join(input_files)))
    add_stmts, _, _, _, _, _ = parse_annotations(input_files)

    print('Usage:\n  %s %s input1.hpp input2.hpp ... output.hpp' % (sys.executable, sys.argv[0], ))

    sys.exit(1)

input_files, output_filename = sys.argv[1:-1], sys.argv[-1]
cpu_output_header = os.path.splitext(output_filename)[0] + '_cpu.hpp'
gpu_output_header = os.path.splitext(output_filename)[0] + '_gpu.hpp'
assert input_files, sys.argv

add_stmts, cpu_template_functions, gpu_template_functions, cpu_address_expressions, gpu_address_expressions, cond_fns = parse_annotations(sys.argv[1:-1])

canonical_input_files = [input_file[input_file.find("/QueryEngine/") + 1:] for input_file in input_files]
header_includes = ['#include "' + canonical_input_file + '"' for canonical_input_file in canonical_input_files]

# Split up calls to TableFunctionsFactory::add() into chunks
ADD_FUNC_CHUNK_SIZE = 100

def add_method(i, chunk):
	return '''
  NO_OPT_ATTRIBUTE void add_table_functions_%d() const {
    %s
  }
''' % (i, '\n    '.join(chunk))

def add_methods(add_stmts):
	chunks = [ add_stmts[n:n+ADD_FUNC_CHUNK_SIZE] for n in range(0, len(add_stmts), ADD_FUNC_CHUNK_SIZE) ]
	return [ add_method(i,chunk) for i,chunk in enumerate(chunks) ]

def call_methods(add_stmts):
	quot, rem = divmod(len(add_stmts), ADD_FUNC_CHUNK_SIZE)
	return [ 'add_table_functions_%d();' % (i) for i in range(quot + int(0 < rem)) ]

content = '''
/*
  This file is generated by %s. Do no edit!
*/

#include "QueryEngine/TableFunctions/TableFunctionsFactory.h"
%s

/*
  Include the UDTF template initiations:
*/
#include "TableFunctionsFactory_init_cpu.hpp"

// volatile+noinline prevents compiler optimization
#ifdef _WIN32
__declspec(noinline)
#else
 __attribute__((noinline))
#endif
volatile
bool avoid_opt_address(void *address) {
  return address != nullptr;
}

bool functions_exist() {
    bool ret = true;

    ret &= (%s);

    return ret;
}

extern bool g_enable_table_functions;

namespace table_functions {

std::once_flag init_flag;

#if defined(__clang__)
#define NO_OPT_ATTRIBUTE __attribute__((optnone))

#elif defined(__GNUC__) || defined(__GNUG__)
#define NO_OPT_ATTRIBUTE __attribute((optimize("O0")))

#elif defined(_MSC_VER)
#define NO_OPT_ATTRIBUTE

#endif

#if defined(_MSC_VER)
#pragma optimize("", off)
#endif

struct AddTableFunctions {
%s
  NO_OPT_ATTRIBUTE void operator()() {
    %s
  }
};

void TableFunctionsFactory::init() {
  if (!g_enable_table_functions) {
    return;
  }

  if (!functions_exist()) {
    UNREACHABLE();
    return;
  }

  std::call_once(init_flag, AddTableFunctions{});
}
#if defined(_MSC_VER)
#pragma optimize("", on)
#endif

// conditional check functions
%s

}  // namespace table_functions

''' % (sys.argv[0],
        '\n'.join(header_includes),
        ' &&\n'.join(cpu_address_expressions),
        ''.join(add_methods(add_stmts)),
        '\n    '.join(call_methods(add_stmts)),
        ''.join(cond_fns))

header_content = '''
/*
  This file is generated by %s. Do no edit!
*/
%s

%s
'''

dirname = os.path.dirname(output_filename)

if dirname and not os.path.exists(dirname):
    try:
        os.makedirs(dirname)
    except OSError as e:
        import errno
        if e.errno != errno.EEXIST:
            raise

f = open(output_filename, 'w')
f.write(content)
f.close()

f = open(cpu_output_header, 'w')
f.write(header_content % (sys.argv[0], '\n'.join(header_includes), '\n'.join(cpu_template_functions)))
f.close()

f = open(gpu_output_header, 'w')
f.write(header_content % (sys.argv[0], '\n'.join(header_includes), '\n'.join(gpu_template_functions)))
f.close()
