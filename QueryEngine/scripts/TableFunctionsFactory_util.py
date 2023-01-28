from collections import namedtuple


OutputBufferSizeTypes = '''
kConstant, kUserSpecifiedConstantParameter, kUserSpecifiedRowMultiplier,
kTableFunctionSpecifiedParameter, kPreFlightParameter
'''.strip().replace(' ', '').split(',')

SupportedAnnotations = '''
input_id, name, fields, require, range, default
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


Signature = namedtuple('Signature', ['name', 'inputs', 'outputs',
                                     'input_annotations', 'output_annotations',
                                     'function_annotations', 'sizer'])


def tostring(obj):
    return obj.tostring()


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
