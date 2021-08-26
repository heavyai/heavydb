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
    RowMultiplier<i>, ConstantParameter<i>, Constant<i>
  where i is literal integer

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
Signature = namedtuple('Signature', ['name', 'inputs', 'outputs', 'input_annotations', 'output_annotations'])

ExtArgumentTypes = ''' Int8, Int16, Int32, Int64, Float, Double, Void, PInt8, PInt16,
PInt32, PInt64, PFloat, PDouble, PBool, Bool, ArrayInt8, ArrayInt16,
ArrayInt32, ArrayInt64, ArrayFloat, ArrayDouble, ArrayBool, GeoPoint,
GeoLineString, Cursor, GeoPolygon, GeoMultiPolygon, ColumnInt8,
ColumnInt16, ColumnInt32, ColumnInt64, ColumnFloat, ColumnDouble,
ColumnBool, ColumnTextEncodingDict, TextEncodingNone, TextEncodingDict,
ColumnListInt8, ColumnListInt16, ColumnListInt32, ColumnListInt64,
ColumnListFloat, ColumnListDouble, ColumnListBool, ColumnListTextEncodingDict'''.strip().replace(' ', '').replace('\n', '').split(',')

OutputBufferSizeTypes = '''
kConstant, kUserSpecifiedConstantParameter, kUserSpecifiedRowMultiplier, kTableFunctionSpecifiedParameter
'''.strip().replace(' ', '').split(',')

SupportedAnnotations = '''
input_id, name
'''.strip().replace(' ', '').split(',')

translate_map = dict(
    Constant='kConstant',
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
          'TextEncodingDict']:
    translate_map[t.lower()] = t
    if t.startswith('Int'):
        translate_map[t.lower() + '_t'] = t


class Bracket:
    """Holds a `NAME<ARGS>`-like structure.
    """

    def __init__(self, name, args=None):
        assert isinstance(name, str)
        assert isinstance(args, tuple) or args is None, args
        self.name = name
        self.args = args

    def __repr__(self):
        return 'Bracket(%r, %r)' % (self.name, self.args)

    def __str__(self):
        if not self.args:
            return self.name
        return '%s<%s>' % (self.name, ', '.join(map(str, self.args)))

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

    def is_column_any(self):
        return self.name.rsplit("::", 1)[-1].startswith('Column')

    def is_column_list(self):
        return self.name.rsplit("::", 1)[-1].startswith('ColumnList')

    def is_column(self):
        return self.name.rsplit("::", 1)[-1].startswith('Column') and not self.is_column_list()

    def is_any_text_encoded_dict(self):
        return self.name.rsplit("::", 1)[-1].endswith('TextEncodedDict')

    def is_column_text_encoded_dict(self):
        return self.name.rsplit("::", 1)[-1] == 'ColumnTextEncodedDict'

    def is_column_list_text_encoded_dict(self):
        return self.name.rsplit("::", 1)[-1] == 'ColumnListTextEncodedDict'

    def is_output_buffer_sizer(self):
        return self.name.rsplit("::", 1)[-1] in OutputBufferSizeTypes

    def is_row_multiplier(self):
        return self.name.rsplit("::", 1)[-1] == 'kUserSpecifiedRowMultiplier'

    def is_user_specified(self):
        # Return True if given argument cannot specified by user
        if self.is_output_buffer_sizer():
            return self.name.rsplit("::", 1)[-1] not in ('kConstant', 'kTableFunctionSpecifiedParameter')
        return True

    def get_cpp_type(self):
        name = self.name.rsplit("::", 1)[-1]
        clsname = None
        if name.startswith('ColumnList'):
            name = name.lstrip('ColumnList')
            clsname = 'ColumnList'
        elif name.startswith('Column'):
            name = name.lstrip('Column')
            clsname = 'Column'
        if name.startswith('Bool'):
            ctype = name.lower()
        elif name.startswith('Int'):
            ctype = name.lower() + '_t'
        elif name in ['Double', 'Float']:
            ctype = name.lower()
        elif name == 'TextEncodingDict':
            ctype = name
        else:
            raise NotImplementedError(self)
        if clsname is None:
            return ctype
        return '%s<%s>' % (clsname, ctype)

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
            rest = typ[i+1:-1].strip()
            while rest:
                i = find_comma(rest)
                if i == -1:
                    a, rest = rest, ''
                else:
                    a, rest = rest[:i].rstrip(), rest[i+1:].lstrip()
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
    # `!' is used to separate the UDTF signature and the expected result
    return line.endswith(',') or line.endswith('->') or line.endswith('!')
# fmt: on


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
        else:
            self.raise_tokenize_error()
        self.advance()

    def consume_whitespace(self):
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
            'Could not match char "%s" at pos %d on line\n%s' % (char, curr, self.line)
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
        if udtf_node.templates:
            templates = ", ".join([t.accept(self) for t in udtf_node.templates])
            return "%s(%s) -> %s, %s" % (name, inputs, outputs, templates)
        else:
            return "%s(%s) -> %s" % (name, inputs, outputs)

    def visit_template_node(self, template_node):
        # T=[T1, T2, ..., TN]
        key = template_node.key
        types = ['"%s"' % typ for typ in template_node.types]
        return "%s=[%s]" % (key, ", ".join(types))

    def visit_annotation_node(self, annotation_node):
        # key=value
        key = annotation_node.key
        value = annotation_node.value
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

        udtfs = []

        d = dict([(node.key, node.types) for node in udtf_node.templates])
        name = udtf_node.name

        for product in product_dict(**d):
            self.mapping_dict = product
            inputs = [input_arg.accept(self) for input_arg in udtf_node.inputs]
            outputs = [output_arg.accept(self) for output_arg in udtf_node.outputs]
            udtfs.append(UdtfNode(name, inputs, outputs, None, udtf_node.line))
            self.mapping_dict = {}

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


class NormalizeTransformer(AstTransformer):
    def visit_udtf_node(self, udtf_node):
        """
        * Add default_input_id to Column(List)<TextEncodingDict> without one
        """
        udtf_node = super(type(self), self).visit_udtf_node(udtf_node)

        # add default input_id
        default_input_id = None
        for idx, t in enumerate(udtf_node.inputs):
            t = t.type
            if not isinstance(t, ComposedNode):
                continue
            if t.is_column_text_encoding_dict():
                default_input_id = AnnotationNode('input_id', 'args<%s>' % (idx,))
                break
            elif t.is_column_list_text_encoding_dict():
                default_input_id = AnnotationNode('input_id', 'args<%s, 0>' % (idx,))
                break

        for t in udtf_node.outputs:
            if isinstance(t.type, ComposedNode) and t.type.is_any_text_encoding_dict():
                for a in t.annotations:
                    if a.key == 'input_id':
                        break
                else:
                    if not default_input_id:
                        raise TypeError('Cannot parse line "%s".\n'
                                        'Missing TextEncodingDict input?' %
                                        (udtf_node.line))
                    t.annotations.append(default_input_id)

        return udtf_node

    def visit_primitive_node(self, primitive_node):
        """
        * Rename nodes using translate_map as dictionary
            int -> Int32
            float -> Float

        * Fix kUserSpecifiedRowMultiplier without a pos arg
        """
        t = primitive_node.type

        if primitive_node.is_output_buffer_sizer():
            pos = PrimitiveNode(str(primitive_node.get_parent(ArgNode).arg_pos + 1))
            node = ComposedNode(t, inner=[pos])
            return node

        return primitive_node.copy(translate_map.get(t, t))


class SignatureTransformer(AstTransformer):
    def visit_udtf_node(self, udtf_node):
        name = udtf_node.name
        inputs = []
        input_annotations = []
        outputs = []
        output_annotations = []

        for i in udtf_node.inputs:
            inp, anns = i.accept(self)
            assert not isinstance(inp, list)  # list inp is not tested
            inputs += inp if isinstance(inp, list) else [inp]
            input_annotations.append(anns)

        for o in udtf_node.outputs:
            out, anns = o.accept(self)
            assert not isinstance(out, list)  # list out is not tested
            outputs += out if isinstance(out, list) else [out]
            output_annotations.append(anns)

        assert len(inputs) == len(input_annotations), (inputs, input_annotations)
        assert len(outputs) == len(output_annotations), (outputs, output_annotations)

        return Signature(name, inputs, outputs, input_annotations, output_annotations)

    def visit_arg_node(self, arg_node):
        t = arg_node.type.accept(self)
        anns = [a.accept(self) for a in arg_node.annotations]
        return t, anns

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
        return Bracket(translate_map.get(t, t))

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

    def __init__(self, name, inputs, outputs, templates, line):
        """
        Parameters
        ----------
        name : str
        inputs : list[ArgNode]
        outputs : list[ArgNode]
        templates : Optional[list[TemplateNode]]
        line: str
        """
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.templates = templates
        self.line = line

    def accept(self, visitor):
        return visitor.visit_udtf_node(self)

    def __str__(self):
        name = self.name
        inputs = [str(i) for i in self.inputs]
        outputs = [str(o) for o in self.outputs]
        if self.templates:
            templates = [str(t) for t in self.templates]
            return "UDFT: %s (%s) -> %s, %s" % (name, inputs, outputs, templates)
        else:
            return "UDFT: %s (%s) -> %s" % (name, inputs, outputs)

    def __iter__(self):
        for i in self.inputs:
            yield i
        for o in self.outputs:
            yield o
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
            anns = "| ".join([str(a) for a in self.annotations])
            return "ArgNode(%s %s)" % (t, anns)
        return "ArgNode(%s)" % (t)

    def __iter__(self):
        yield self.type
        for a in self.annotations:
            yield a

    __repr__ = __str__


class TypeNode(Node):
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
        return "Primitive(%s)" % (self.type)

    def is_text_encoding_dict(self):
        return self.type == 'TextEncodingDict'

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

    def is_column_text_encoding_dict(self):
        return self.is_column() and self.inner[0].is_text_encoding_dict()

    def is_column_list_text_encoding_dict(self):
        return self.is_column_list() and self.inner[0].is_text_encoding_dict()

    def is_any_text_encoding_dict(self):
        return self.inner[0].is_text_encoding_dict()

    __repr__ = __str__


class AnnotationNode(Node):

    def __init__(self, key, value):
        """
        Parameters
        ----------
        key : str
        value : str
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

    def parse_udtf(self):
        """fmt: off

        udtf: IDENTIFIER "(" args ")" "->" args ("," templates)?

        fmt: on
        """
        name = self.parse_identifier()
        self.expect(Token.LPAR)  # (
        input_args = self.parse_args()
        self.expect(Token.RPAR)  # )
        self.expect(Token.RARROW)
        output_args = self.parse_args()

        templates = None
        if not self.is_at_end() and self.match(Token.COMMA):
            self.consume(Token.COMMA)
            templates = self.parse_templates()

        # set arg_pos
        i = 0
        for arg in input_args:
            arg.arg_pos = i
            arg.kind = "input"
            i += arg.type.cursor_length() if arg.type.is_cursor() else 1

        for i, arg in enumerate(output_args):
            arg.arg_pos = i
            arg.kind = "output"

        return UdtfNode(name, input_args, output_args, templates, self.line)

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

        composed: IDENTIFIER "<" type ("," type)* ">"

        fmt: on
        """
        idtn = self.parse_identifier()
        self.consume(Token.LESS)
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

        fmt: on
        """
        if self.match(Token.IDENTIFIER):
            lexeme = self.parse_identifier()
        elif self.match(Token.NUMBER):
            lexeme = self.parse_number()
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

        fmt: on
        """
        key = self.parse_identifier()
        self.consume(Token.EQUAL)
        value = self.parse_identifier()
        if not self.is_at_end() and self.match(Token.LESS):
            self.consume(Token.LESS)
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

    def parse_number(self):
        """ fmt: off

        NUMBER: [0-9]+

        fmt: on
        """
        token = self.consume(Token.NUMBER)
        return token.lexeme

    def parse(self):
        """fmt: off

        udtf: IDENTIFIER "(" args ")" "->" args ("," templates)?

        args: arg ("," arg)*

        arg: type IDENTIFIER? ("|" annotation)*

        type: composed
            | primitive

        composed: IDENTIFIER "<" type ("," type)* ">"
        primitive: IDENTIFIER
                 | NUMBER

        annotation: IDENTIFIER "=" IDENTIFIER ("<" NUMBER ("," NUMBER) ">")?

        templates: template ("," template)
        template: IDENTIFIER "=" "[" IDENTIFIER ("," IDENTIFIER)* "]"

        IDENTIFIER: [A-Za-z_][A-Za-z0-9_]*
        NUMBER: [0-9]+

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
            line = last_line + line
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
        if '!' in line:
            line, expected_result = line.split('!', 1)
            expected_result = expected_result.strip().split('!')
            expected_result = list(map(lambda s: s.strip(), expected_result))

        ast = Parser(line).parse()

        if expected_result is not None:
            # Template transformer expands templates into multiple lines
            result = Pipeline(TemplateTransformer, NormalizeTransformer, AstPrinter)(ast)
            assert set(result) == set(expected_result), "\n\tresult: %s != \n\texpected: %s" % (
                result,
                expected_result,
            )

        signature = Pipeline(TemplateTransformer,
                             NormalizeTransformer,
                             SignatureTransformer)(ast)
        signatures.extend(signature)

    return signatures


def build_template_function_call(caller, callee, input_types, output_types):
    # caller calls callee

    def format_cpp_type(cpp_type, idx, is_input=True):
        # Perhaps integrate this to Bracket?
        col_typs = ('Column', 'ColumnList')
        idx = str(idx)
        # TODO: use name in annotations when present?
        arg_name = 'input' + idx if is_input else 'out' + idx
        const = 'const ' if is_input else ''

        if any(cpp_type.startswith(ct) for ct in col_typs):
            return '%s%s& %s' % (const, cpp_type, arg_name), arg_name
        else:
            return '%s %s' % (cpp_type, arg_name), arg_name

    input_cpp_args = []
    output_cpp_args = []
    arg_names = []

    for idx, input_type in enumerate(input_types):
        cpp_type = input_type.get_cpp_type()
        cpp_arg, arg_name = format_cpp_type(cpp_type, idx)
        input_cpp_args.append(cpp_arg)
        arg_names.append(arg_name)

    for idx, output_type in enumerate(output_types):
        cpp_type = output_type.get_cpp_type()
        cpp_arg, arg_name = format_cpp_type(cpp_type, idx, is_input=False)
        output_cpp_args.append(cpp_arg)
        arg_names.append(arg_name)

    args = ', '.join(input_cpp_args + output_cpp_args)
    arg_names = ', '.join(arg_names)

    template = ("EXTENSION_NOINLINE int32_t\n"
                "%s(%s) {\n"
                "    return %s(%s);\n"
                "}\n") % (caller, args, callee, arg_names)
    return template


def format_annotations(annotations_):
    s = "std::vector<std::map<std::string, std::string>>{"
    s += ', '.join(('{' + ', '.join('{"%s", "%s"}' % (k, v) for k, v in a) + '}') for a in annotations_)
    s += "}"
    return s


def is_template_function(sig):
    i = sig.name.rfind('_template')
    return i >= 0 and '__' in sig.name[:i+1]


def parse_annotations(input_files):

    counter = 0

    add_stmts = []
    template_functions = []

    for input_file in input_files:
        for sig in find_signatures(input_file):

            # Compute sql_types, input_types, and sizer
            sql_types_ = []
            input_types_ = []
            input_annotations = []
            sizer = None
            for t, annot in zip(sig.inputs, sig.input_annotations):
                if t.is_output_buffer_sizer():
                    if t.is_user_specified():
                        sql_types_.append(Bracket.parse('int32').normalize(kind='input'))
                        input_types_.append(sql_types_[-1])
                        input_annotations.append(annot)
                    assert sizer is None  # exactly one sizer argument is allowed
                    assert len(t.args) == 1, t
                    sizer = 'TableFunctionOutputRowSizer{OutputBufferSizeType::%s, %s}' % (t.name, t.args[0])
                elif t.name == 'Cursor':
                    for t_ in t.args:
                        input_types_.append(t_)
                    input_annotations.append(annot)
                    sql_types_.append(Bracket('Cursor', args=()))
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
                sizer = 'TableFunctionOutputRowSizer{OutputBufferSizeType::%s, %s}' % (name, idx)

            assert sizer is not None
            ns_output_types = tuple([a.apply_namespace(ns='ExtArgumentType') for a in sig.outputs])
            ns_input_types = tuple([t.apply_namespace(ns='ExtArgumentType') for t in input_types_])
            ns_sql_types = tuple([t.apply_namespace(ns='ExtArgumentType') for t in sql_types_])

            input_types = 'std::vector<ExtArgumentType>{%s}' % (', '.join(map(str, ns_input_types)))
            output_types = 'std::vector<ExtArgumentType>{%s}' % (', '.join(map(str, ns_output_types)))
            sql_types = 'std::vector<ExtArgumentType>{%s}' % (', '.join(map(str, ns_sql_types)))
            annotations = format_annotations(input_annotations + sig.output_annotations)

            if is_template_function(sig):
                name = sig.name + '_' + str(counter)
                counter += 1
                t = build_template_function_call(name, sig.name, input_types_, sig.outputs)
                template_functions.append(t)
                add = 'TableFunctionsFactory::add("%s", %s, %s, %s, %s, %s);' % (name, sizer, input_types, output_types, sql_types, annotations)
                add_stmts.append(add)
            else:
                add = 'TableFunctionsFactory::add("%s", %s, %s, %s, %s, %s);' % (sig.name, sizer, input_types, output_types, sql_types, annotations)
                add_stmts.append(add)

    return add_stmts, template_functions


if len(sys.argv) < 3:

    input_files = [os.path.join(os.path.dirname(__file__), 'test_udtf_signatures.hpp')]
    print('Running tests from %s' % (', '.join(input_files)))
    add_stmts, template_functions = parse_annotations(input_files)
    print('Usage:\n  %s %s input1.hpp input2.hpp ... output.hpp' % (sys.executable, sys.argv[0], ))

    sys.exit(1)

input_files, output_filename = sys.argv[1:-1], sys.argv[-1]
assert input_files, sys.argv

add_stmts, template_functions = parse_annotations(sys.argv[1:-1])

content = '''
/*
  This file is generated by %s. Do no edit!
*/

#include "QueryEngine/TableFunctions/TableFunctionsFactory.h"
#include "QueryEngine/TableFunctions/TableFunctions.hpp"
#include "QueryEngine/OmniSciTypes.h"

extern bool g_enable_table_functions;

namespace table_functions {

std::once_flag init_flag;

void TableFunctionsFactory::init() {
  if (!g_enable_table_functions) {
    return;
  }
  std::call_once(init_flag, []() {
    %s
  });
}

%s

}  // namespace table_functions
''' % (sys.argv[0], '\n    '.join(add_stmts), '\n'.join(template_functions))


dirname = os.path.dirname(output_filename)
if dirname and not os.path.exists(dirname):
    os.makedirs(dirname)

f = open(output_filename, 'w')
f.write(content)
f.close()
