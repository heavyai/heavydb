"""Given a list of input files, scan for lines containing UDTF
specification statements in the following form:

  UDTF: function_name(<arguments>) -> <output column types>

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
"""
# Author: Pearu Peterson
# Created: January 2021

import os
import re
import sys
from collections import namedtuple

Signature = namedtuple('Signature', ['name', 'inputs', 'outputs', 'line'])

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


_is_int = re.compile(r'\d+').match


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


def find_signatures(input_file):
    """Returns a list of parsed UDTF signatures.
    """

    def get_function_name(line):
        return line.split('(')[0]

    def get_types_and_annotations(line):
        """Line is a comma separated string of types.
        """
        rest = line.strip()
        types, annotations = [], []
        while rest:
            i = find_comma(rest)
            if i == -1:
                type_annot, rest = rest, ''
            else:
                type_annot, rest = rest[:i].rstrip(), rest[i+1:].lstrip()
            if '|' in type_annot:
                typ, annots = type_annot.split('|', 1)
                typ, annots = typ.rstrip(), annots.lstrip().split('|')
            else:
                typ, annots = type_annot, []
            types.append(typ)
            pairs = []
            for annot in annots:
                label, value = annot.strip().split('=', 1)
                label, value = label.rstrip(), value.lstrip()
                pairs.append((label, value))
            annotations.append(pairs)
        return types, annotations

    def get_input_types_and_annotations(line):
        start = line.rfind('(') + 1
        end = line.find(')')
        assert -1 not in [start, end], line
        return get_types_and_annotations(line[start:end])

    def get_output_types_and_annotations(line):
        start = line.rfind('->') + 2
        end = len(line)
        assert -1 not in [start, end], line
        return get_types_and_annotations(line[start:end])

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
            expected_result = expected_result.strip()

        name = get_function_name(line)
        input_types, input_annotations = get_input_types_and_annotations(line)
        output_types, output_annotations = get_output_types_and_annotations(line)

        input_types = tuple([Bracket.parse(typ).normalize(kind='input') for typ in input_types])
        output_types = tuple([Bracket.parse(typ).normalize(kind='output') for typ in output_types])

        # Apply default sizer
        has_sizer = False
        consumed_nargs = 0
        for i, t in enumerate(input_types):
            if t.is_output_buffer_sizer():
                has_sizer = True
                if t.is_row_multiplier():
                    if not t.args:
                        t.args = Bracket.parse('RowMultiplier<%s>' % (consumed_nargs + 1)).args
            elif t.is_cursor():
                consumed_nargs += len(t.args)
            else:
                consumed_nargs += 1
        if not has_sizer:
            t = Bracket.parse('kTableFunctionSpecifiedParameter<1>')
            input_types += (t,)

        # Apply default input_id to output TextEncodedDict columns
        default_input_id = None
        for i, t in enumerate(input_types):
            if t.is_column_text_encoded_dict():
                default_input_id = 'args<%s>' % (i,)
                break
            elif t.is_column_list_text_encoded_dict():
                default_input_id = 'args<%s, 0>' % (i,)
                break
        for t, annots in zip(output_types, output_annotations):
            if t.is_any_text_encoded_dict():
                has_input_id = False
                for a in annots:
                    if a[0] == 'input_id':
                        has_input_id = True
                        break
                if not has_input_id:
                    assert default_input_id is not None
                    annots.append(('input_id', default_input_id))

        result = name + '('
        result += ', '.join([' | '.join([str(t)] + [k + '=' + v for k, v in a]) for t, a in zip(input_types, input_annotations)])
        result += ') -> '
        result += ', '.join([' | '.join([str(t)] + [k + '=' + v for k, v in a]) for t, a in zip(output_types, output_annotations)])

        if expected_result is not None:
            assert result == expected_result, (result, expected_result)
            if 1:
                # Make sure that we have stable parsing result
                line = result
                name = get_function_name(line)
                input_types, input_annotations = get_input_types_and_annotations(line)
                output_types, output_annotations = get_output_types_and_annotations(line)
                input_types = tuple([Bracket.parse(typ).normalize(kind='input') for typ in input_types])
                output_types = tuple([Bracket.parse(typ).normalize(kind='output') for typ in output_types])
                result2 = name + '('
                result2 += ', '.join([' | '.join([str(t)] + [k + '=' + v for k, v in a]) for t, a in zip(input_types, input_annotations)])
                result2 += ') -> '
                result2 += ', '.join([' | '.join([str(t)] + [k + '=' + v for k, v in a]) for t, a in zip(output_types, output_annotations)])
                assert result == result2, (result, result2)
        signatures.append(Signature(name, input_types, output_types, input_annotations, output_annotations))

    return signatures


def is_template_function(sig):
    return '_template' in sig.name


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


def parse_annotations(input_files):

    counter = 0

    add_stmts = []
    template_functions = []

    for input_file in input_files:
        for sig in find_signatures(input_file):

            # Compute sql_types, input_types, and sizer
            sql_types_ = []
            input_types_ = []
            sizer = None
            for t in sig.inputs:
                if t.is_output_buffer_sizer():
                    if t.is_user_specified():
                        sql_types_.append(Bracket.parse('int32').normalize(kind='input'))
                        input_types_.append(sql_types_[-1])
                    assert sizer is None  # exactly one sizer argument is allowed
                    assert len(t.args) == 1, t
                    sizer = 'TableFunctionOutputRowSizer{OutputBufferSizeType::%s, %s}' % (t.name, t.args[0])
                elif t.name == 'Cursor':
                    for t_ in t.args:
                        input_types_.append(t_)
                    sql_types_.append(Bracket('Cursor', args=()))
                else:
                    input_types_.append(t)
                    if t.is_column_any():
                        # XXX: let Bracket handle mapping of column to cursor(column)
                        sql_types_.append(Bracket('Cursor', args=()))
                    else:
                        sql_types_.append(t)

            if sizer is None:
                name = 'kTableFunctionSpecifiedParameter'
                idx = 1  # this sizer is not actually materialized in the UDTF
                sizer = 'TableFunctionOutputRowSizer{OutputBufferSizeType::%s, %s}'  % (name, idx)

            assert sizer is not None

            ns_output_types = tuple([a.apply_namespace(ns='ExtArgumentType') for a in sig.outputs])
            ns_input_types = tuple([t.apply_namespace(ns='ExtArgumentType') for t in input_types_])
            ns_sql_types = tuple([t.apply_namespace(ns='ExtArgumentType') for t in sql_types_])

            input_types = 'std::vector<ExtArgumentType>{%s}' % (', '.join(map(str, ns_input_types)))
            output_types = 'std::vector<ExtArgumentType>{%s}' % (', '.join(map(str, ns_output_types)))
            sql_types = 'std::vector<ExtArgumentType>{%s}' % (', '.join(map(str, ns_sql_types)))
            annotations = format_annotations(sig.input_annotations + sig.output_annotations)


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

