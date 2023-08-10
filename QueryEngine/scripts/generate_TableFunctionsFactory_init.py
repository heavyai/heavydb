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
- default: to specify a default value for an argument (scalar only)

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
import warnings

import TableFunctionsFactory_transformers as transformers
import TableFunctionsFactory_parser as parser
import TableFunctionsFactory_declbracket as declbracket
import TableFunctionsFactory_util as util
import TableFunctionsFactory_linker as linker


# fmt: off
separator = '$=>$'

def line_is_incomplete(line):
    # TODO: try to parse the line to be certain about completeness.
    # `$=>$' is used to separate the UDTF signature and the expected result
    return line.endswith(',') or line.endswith('->') or line.endswith(separator) or line.endswith('|')


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

        ast = parser.Parser(line).parse()

        if expected_result is not None:
            # Treat warnings as errors so that one can test TransformeWarnings
            warnings.filterwarnings("error")

            # Template transformer expands templates into multiple lines
            try:
                result = transformers.Pipeline(
                    transformers.TemplateTransformer,
                    transformers.AmbiguousSignatureCheckTransformer,
                    transformers.FieldAnnotationTransformer,
                    transformers.TextEncodingDictTransformer,
                    transformers.DefaultValueAnnotationTransformer,
                    transformers.SupportedAnnotationsTransformer,
                    transformers.RangeAnnotationTransformer,
                    transformers.CursorAnnotationTransformer,
                    transformers.FixRowMultiplierPosArgTransformer,
                    transformers.RenameNodesTransformer,
                    transformers.AstPrinter)(ast)
            except (transformers.TransformerException, transformers.TransformerWarning) as msg:
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
            signature = transformers.Pipeline(
                transformers.TemplateTransformer,
                transformers.AmbiguousSignatureCheckTransformer,
                transformers.FieldAnnotationTransformer,
                transformers.TextEncodingDictTransformer,
                transformers.DefaultValueAnnotationTransformer,
                transformers.SupportedAnnotationsTransformer,
                transformers.RangeAnnotationTransformer,
                transformers.CursorAnnotationTransformer,
                transformers.FixRowMultiplierPosArgTransformer,
                transformers.RenameNodesTransformer,
                transformers.DeclBracketTransformer)(ast)

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
        if isinstance(typ, declbracket.Declaration):
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
        if k == 'require' or k == 'default' and v[0] == "\"":
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
                sizer = declbracket.Bracket('kPreFlightParameter', (expr,))

            uses_manager = False
            for i, (t, annot) in enumerate(zip(sig.inputs, sig.input_annotations)):
                if t.is_output_buffer_sizer():
                    if t.is_user_specified():
                        sql_types_.append(declbracket.Bracket.parse('int32').normalize(kind='input'))
                        input_types_.append(sql_types_[-1])
                        input_annotations.append(annot)
                    assert sizer is None  # exactly one sizer argument is allowed
                    assert len(t.args) == 1, t
                    sizer = t
                elif t.name == 'Cursor':
                    for t_ in t.args:
                        input_types_.append(t_)
                    input_annotations.append(annot)
                    sql_types_.append(declbracket.Bracket('Cursor', args=()))
                elif t.name == 'TableFunctionManager':
                    if i != 0:
                        raise ValueError('{} must appear as a first argument of {}, but found it at position {}.'.format(t, sig.name, i))
                    uses_manager = True
                else:
                    input_types_.append(t)
                    input_annotations.append(annot)
                    if t.is_column_any():
                        # XXX: let Bracket handle mapping of column to cursor(column)
                        sql_types_.append(declbracket.Bracket('Cursor', args=()))
                    else:
                        sql_types_.append(t)

            if sizer is None:
                name = 'kTableFunctionSpecifiedParameter'
                idx = 1  # this sizer is not actually materialized in the UDTF
                sizer = declbracket.Bracket(name, (idx,))

            assert sizer is not None
            ns_output_types = tuple([a.apply_namespace(ns='ExtArgumentType') for a in sig.outputs])
            ns_input_types = tuple([t.apply_namespace(ns='ExtArgumentType') for t in input_types_])
            ns_sql_types = tuple([t.apply_namespace(ns='ExtArgumentType') for t in sql_types_])

            sig.function_annotations.append(('uses_manager', str(uses_manager).lower()))

            input_types = 'std::vector<ExtArgumentType>{%s}' % (', '.join(map(util.tostring, ns_input_types)))
            output_types = 'std::vector<ExtArgumentType>{%s}' % (', '.join(map(util.tostring, ns_output_types)))
            sql_types = 'std::vector<ExtArgumentType>{%s}' % (', '.join(map(util.tostring, ns_sql_types)))
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

add_stmts = []
cpu_template_functions = []
gpu_template_functions = []
cpu_address_expressions = []
gpu_address_expressions = []
cond_fns = []

canonical_input_files = [input_file[input_file.find("/QueryEngine/") + 1:] for input_file in input_files]
header_file = ['#include "' + canonical_input_file + '"' for canonical_input_file in canonical_input_files]

dirname = os.path.dirname(output_filename)

if dirname and not os.path.exists(dirname):
    try:
        os.makedirs(dirname)
    except OSError as e:
        import errno
        if e.errno != errno.EEXIST:
            raise


for input_file in input_files:
    stmts, cpu_fns, gpu_fns, cpu_addr, gpu_addr, cond_funcs = parse_annotations([input_file])

    add_stmts.extend(stmts)
    cpu_template_functions.extend(cpu_fns)
    gpu_template_functions.extend(gpu_fns)
    cpu_address_expressions.extend(cpu_addr)
    gpu_address_expressions.extend(gpu_addr)
    cond_fns.extend(cond_funcs)

    header_file = input_file[input_file.find("/QueryEngine/") + 1:]

    add_tf_generated_files = linker.GenerateAddTableFunctionsFiles(dirname, stmts,
                                                                   header_file)
    if add_tf_generated_files.should_generate_files():
        add_tf_generated_files.generate_files()

    if len(cpu_fns):
        cpu_generated_files = linker.GenerateTemplateFiles(dirname, cpu_fns,
                                                           header_file, 'cpu')
        cpu_generated_files.generate_files()

    if len(gpu_fns):
        gpu_generated_files = linker.GenerateTemplateFiles(dirname, gpu_fns,
                                                           header_file, 'gpu')
        gpu_generated_files.generate_files()


def call_methods(add_stmts):
    n_add_funcs = linker.GenerateAddTableFunctionsFiles.get_num_generated_files()
    return [ 'table_functions::add_table_functions_%d();' % (i) for i in range(n_add_funcs+1) ]


content = '''
/*
  This file is generated by %s. Do no edit!
*/

#include "QueryEngine/TableFunctions/TableFunctionsFactory.h"


/*
  Include the UDTF template initiations:
*/
%s

// volatile+noinline prevents compiler optimization
#ifdef _WIN32
__declspec(noinline)
#else
 __attribute__((noinline))
#endif

#ifndef NO_OPT_ATTRIBUTE
#if defined(__clang__)
#define NO_OPT_ATTRIBUTE __attribute__((optnone))

#elif defined(__GNUC__) || defined(__GNUG__)
#define NO_OPT_ATTRIBUTE __attribute__((optimize("O0")))

#elif defined(_MSC_VER)
#define NO_OPT_ATTRIBUTE

#endif
#endif

#if defined(_MSC_VER)
#pragma optimize("", off)
#endif

volatile
NO_OPT_ATTRIBUTE bool avoid_opt_address(void *address) {
  return address != nullptr;
}

NO_OPT_ATTRIBUTE bool functions_exist() {
    bool ret = true;

    ret &= (%s);

    return ret;
}

extern bool g_enable_table_functions;

extern bool functions_exist_geo_column();

// Each table function initialization module needs its own AddTableFunctions struct definition,
// otherwise, when calling an initialization function at runtime, symbol name conflicts will
// cause the wrong struct to be instantiated.
namespace {
struct AddTableFunctions {
  NO_OPT_ATTRIBUTE void operator()() {
    %s
  }
};
} // anonymous namespace

namespace table_functions {

// Each table function initialization module should have its own init flag
static std::once_flag init_flag;

static const char filename[] = __FILE__;

template<const char *filename>
void TableFunctionsFactory::init() {
  if (!g_enable_table_functions) {
    return;
  }

  if (!functions_exist() && !functions_exist_geo_column()) {
    UNREACHABLE();
    return;
  }

  std::call_once(init_flag, AddTableFunctions{});
}

extern "C" void init_table_functions() {
    TableFunctionsFactory::init<filename>();
}
#if defined(_MSC_VER)
#pragma optimize("", on)
#endif

// conditional check functions
%s

}  // namespace table_functions

'''

#####

content = content % (
    sys.argv[0],
    '\n'.join(map(lambda x: '#include "%s"' % x, linker.BaseGenerateFiles.generated_header_files())),
    ' &&\n'.join(cpu_address_expressions),
    '\n    '.join(call_methods(add_stmts)),
    ''.join(cond_fns))


if not (os.path.exists(output_filename) and \
        content == linker.get_existing_file_content(output_filename)):
    with open(output_filename, 'w') as f:
        f.write(content)
