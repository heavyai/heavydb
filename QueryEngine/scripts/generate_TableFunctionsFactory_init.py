"""Given a list of input files, scan for lines containing UDTF
specification statements in the following form:

  UDTF: function_name(<arguments>) -> <output column types>

where <arguments> is a comma-separated list of argument types. The
argument types specifications are:

- scalar types:
    Int8, Int16, Int32, Int64, Float, Double, Bool, etc
- column types:
    ColumnInt8, ColumnInt16, ColumnInt32, ColumnInt64, ColumnFloat, ColumnDouble, ColumnBool, etc
- cursor type:
    Cursor<t0, t1, ...>
  where t0, t1 are column types
- output buffer size parameter type:
    RowMultiplier<i>, ConstantParameter<i>, Constant<i>
  where i is literal integer

The output column types is a comma-separated list of column types, see above.

In addition, the following equivalents are suppored:
  Column<T> == ColumnT
  Cursor<T, V, ...> == Cursor<ColumnT, ColumnV, ...>
  int8 == int8_t == Int8, etc
  float == Float, double == Double, bool == Bool
  T == ColumnT for output column types
  RowMultiplier == RowMultiplier<i> where i is the one-based position of the sizer argument
  when no sizer argument is provided, Constant<1> is assumed
"""
# Author: Pearu Peterson
# Created: January 2021

import os
import re
import sys

ExtArgumentTypes = '''
Int8, Int16, Int32, Int64, Float, Double, Void, PInt8, PInt16, PInt32,
PInt64, PFloat, PDouble, PBool, Bool, ArrayInt8, ArrayInt16,
ArrayInt32, ArrayInt64, ArrayFloat, ArrayDouble, ArrayBool, GeoPoint,
GeoLineString, Cursor, GeoPolygon, GeoMultiPolygon, ColumnInt8,
ColumnInt16, ColumnInt32, ColumnInt64, ColumnFloat, ColumnDouble,
ColumnBool, TextEncodingNone, TextEncodingDict8, TextEncodingDict16,
TextEncodingDict32
'''.strip().replace(' ', '').split(',')

OutputBufferSizeTypes = '''
kConstant, kUserSpecifiedConstantParameter, kUserSpecifiedRowMultiplier
'''.strip().replace(' ', '').split(',')

translate_map = dict(
    Constant = 'kConstant',
    ConstantParameter = 'kUserSpecifiedConstantParameter',
    RowMultiplier = 'kUserSpecifiedRowMultiplier',
    UserSpecifiedConstantParameter = 'kUserSpecifiedConstantParameter',
    UserSpecifiedRowMultiplier = 'kUserSpecifiedRowMultiplier',
    short = 'Int16',
    int = 'Int32',
    long = 'Int64',
)
for t in ['Int8', 'Int16', 'Int32', 'Int64', 'Float', 'Double', 'Bool']:
    translate_map[t.lower()] = t
    if t.startswith('Int'):
        translate_map[t.lower() + '_t'] = t


_is_int = re.compile(r'\d+').match

def type_parse(a):
    i = a.find('<')
    if i >= 0:
        assert a.endswith('>')
        n = a[:i]
        n = translate_map.get(n, n)
        if n in OutputBufferSizeTypes:
            v = a[i+1:-1]
            assert _is_int(v)
            return n, v
        if n == 'Cursor':
            lst = []
            for t in map(type_parse, a[i+1:-1].split(',')):
                if 'Column' + t in ExtArgumentTypes:
                    lst.append('Column' + t)
                else:
                    lst.append(t)
            return n, tuple(lst)
        if n == 'Column':
            return n + type_parse(a[i+1:-1])
    else:
        a = translate_map.get(a, a)
        if a in ExtArgumentTypes:
            return a
        if a in OutputBufferSizeTypes:
            return a, None
    raise ValueError('Cannot parse `%s` to ExtArgumentTypes or OutputBufferSizeTypes' % (a,))
        

add_stmts = []    

for input_file in sys.argv[1:-1]:
    for line in open(input_file).readlines():
        line = line.replace(' ', '').strip()
        if not line.startswith('UDTF:'):
            continue
        line = line[5:]
        i = line.find('(')
        j = line.find(')')
        if i == -1 or j == -1:
            sys.stderr.write('Invalid UDTF specification: `%s`. Skipping.\n' % (line))
            continue
        name = line[:i]
        args_line = line[i+1:j]
        outputs = line[j+1:]
        if outputs.startswith('->'):
            outputs = outputs[2:]
        outputs = outputs.split(',')

        args = []
        while args_line:
            i = args_line.find(',')
            if i == -1:
                args.append(args_line)
                break
            j = args_line.find('<')
            k = args_line.find('>')
            if j == -1 or i < j:
                args.append(args_line[:i])
                args_line = args_line[i+1:]
            else:
                assert k != -1
                args.append(args_line[:k+1])
                args_line = args_line[k+1:].lstrip(',')

        input_types = []
        output_types = []
        sql_types = []
        sizer = None
        for i, a in enumerate(args):
            try:
                r = type_parse(a)
            except ValueError as msg:
                raise ValueError('`%s`: %s' % (line, msg))
            if isinstance(r, str) and r.startswith('Column'):
                r = 'Cursor', (r,)
            if isinstance(r, str):
                input_types.append(r)
                sql_types.append(r)
            else:
                n, t = r
                if n in OutputBufferSizeTypes:
                    if n != 'kConstant':
                        input_types.append('ExtArgumentType::Int32')
                        sql_types.append('ExtArgumentType::Int32')
                    if n == 'kUserSpecifiedRowMultiplier':
                        if not t:
                            t = str(i + 1)
                        assert t == str(i+1), 'Expected %s<%s> got %s<%s> from %s' % (n, i+1, n, t, a)
                    assert sizer is None  # exactly one sizer argument is allowed
                    sizer = 'TableFunctionOutputRowSizer{OutputBufferSizeType::%s, %s}' % (n, t)
                else:
                    assert n == 'Cursor', (a, r)
                    for t_ in t:
                        input_types.append('ExtArgumentType::%s' % (t_))
                    sql_types.append('ExtArgumentType::%s' % (n))

        for a in outputs:
            try:
                r = type_parse(a)
            except ValueError as msg:
                raise ValueError('`%s`: %s' % (line, msg))
            assert isinstance(r, str), (a, r)
            if 'Column' + r in ExtArgumentTypes:
                r = 'Column' + r
            output_types.append('ExtArgumentType::%s' % (r))

        if sizer is None:
            sizer = 'TableFunctionOutputRowSizer{OutputBufferSizeType::kConstant, 1}'

        input_types = 'std::vector<ExtArgumentType>{%s}' % (', '.join(input_types))
        output_types = 'std::vector<ExtArgumentType>{%s}' % (', '.join(output_types))
        sql_types = 'std::vector<ExtArgumentType>{%s}' % (', '.join(sql_types)) 
        add = 'TableFunctionsFactory::add("%s", %s, %s, %s, %s);' % (name, sizer, input_types, output_types, sql_types)
        add_stmts.append(add)


content = '''
/*
  This file is generated by %s. Do no edit!
*/

#include "QueryEngine/TableFunctions/TableFunctionsFactory.h"

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

}  // namespace table_functions
''' % (sys.argv[0], '\n    '.join(add_stmts))

output_filename = sys.argv[-1]
dirname = os.path.dirname(output_filename)
if not os.path.exists(dirname):
    os.makedirs(dirname)

f = open(output_filename, 'w')
f.write(content)
f.close()
