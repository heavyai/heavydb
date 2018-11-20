import os
import numpy
import numba
import inspect
import zlib
import itertools
import remotedict
import re

def to_numba_type(t):
    if isinstance(t, numba.types.Type):
        return t
    if t == float or t == inspect._empty:
        return numba.types.float64
    if t == int:
        return numba.types.int64
    if t == bool:
        return numba.types.boolean
    if t == complex:
        return numba.types.complex128
    if issubclass(t, numpy.number):
        t = getattr(numba.types, t.__name__, None)
        if t is not None:
            return t
        raise
    raise NotImplementedError(repr((t, type(t))))

def compact_numba_type(t):
    # follow numpy convention
    if t.name == 'bool': return 't' # `?` not usable in name
    if t.name == 'float64': return 'd'
    if t.name == 'float32': return 'f'
    if t.name == 'int64': return 'l'
    if t.name == 'int32': return 'i'
    if t.name == 'int16': return 'h'
    if t.name == 'int8': return 'b'
    if t.name.startswith('float'):
        return 'f'+str(t.bitwidth//8)
    if t.name.startswith('int'):
        return 'i'+str(t.bitwidth//8)
    if t.name.startswith('uint'):
        return 'u'+str(t.bitwidth//8)
    if t.name.startswith('complex'):
        return 'c'+str(t.bitwidth//8)
    raise NotImplementedError(repr(t))

def compact_prototype(atypes, rtype):
    return ''.join(map(compact_numba_type, atypes)) + \
        '_' + compact_numba_type(rtype)

def c_numba_type(t):
    if t.name == 'float64': return 'double'
    if t.name == 'float32': return 'float'
    if t.name == 'int64': return 'int64_t'
    if t.name == 'int32': return 'int32_t'
    if t.name == 'int16': return 'int16_t'
    if t.name == 'int8': return 'int8_t'
    if t.name == 'bool': return 'bool'
    raise NotImplementedError(repr(t))

def SqlTypeFamily_numba_type(t):
    if t.name == 'bool': return 'SqlTypeFamily.NUMERIC'
    if t.name.startswith('float'): return 'SqlTypeFamily.NUMERIC'
    if t.name.startswith('int'): return 'SqlTypeFamily.NUMERIC'
    if t.name.startswith('uint'): return 'SqlTypeFamily.NUMERIC'
    if t.name.startswith('complex'): return 'SqlTypeFamily.NUMERIC'
    raise NotImplementedError(repr(t))

def SqlTypeName_numba_type(t):
    if t.name == 'bool': return 'SqlTypeFamily.BOOLEAN'
    if t.name == 'float64': return 'SqlTypeName.DOUBLE'
    if t.name == 'float32': return 'SqlTypeName.FLOAT'
    if t.name == 'int64': return 'SqlTypeName.BIGINT'
    if t.name == 'int32': return 'SqlTypeName.INTEGER'
    if t.name == 'int16': return 'SqlTypeName.SMALLINT'
    if t.name == 'int8': return 'SqlTypeName.TINYINT'
    raise NotImplementedError(repr(t))

def make_prototype(func):
    sig = inspect.signature(func)
    atypes = []
    for typ in sig.parameters.values():
        atypes.append(to_numba_type(typ.annotation))
    rtype = to_numba_type(sig.return_annotation)
    prototype = rtype(*atypes)
    return prototype, compact_prototype(atypes, rtype)


class UDF:

    def __init__(self, prototype, func, func_id):
        self.prototype = prototype
        self.func = func
        self.func_id = func_id
        self._cfunc = None
        self._address = None
        self.nargs = len(prototype[0].args)
        
    def __repr__(self):
        return '{}({}, {!r}, {})'.format(
            type(self).__name__, self.prototype, self.func.__name__, self.func_id
        )
        
    def sql(self, *args):
        """
        Parameters
        ----------
        args : tuple
          Specify arguments as SQL expressions

        Returns
        -------
        call : str
          SQL call expression of the given UDF
        """
        if len(args) != self.nargs:
            raise ValueError('expected {} arguments but got {}'
                             .format(self.nargs, len(args)))
        return 'pyudf_{proto}({func_id}, {args})'.format(
            proto = self.prototype[1],
            func_id = self.func_id, args = ', '.join(map(str, args))
        )

    def __call__(self, *args, **kwargs):
        return self.sql(*args, **kwargs)
        #return self.func(*args, **kwargs)

    @property
    def cfunc(self):
        if self._cfunc is None:
            self._cfunc = numba.cfunc(self.prototype[0])(self.func)
        return self._cfunc

    @property
    def address(self):
        address = self._address
        if address is None:
            self._address = address = self.cfunc.address
        return address

class mapd:
    """ Decorator for MapD user-defined functions.
    """

    def __init__(self, host=None, port=None):
        """
        Parameters
        ----------
        host : str
          Specify server host name
        port : int
          Specify server port
        """
        self.host = host
        self.port = port
        
    def __call__(self, func):
        # Compute reference value of a function as a 64 bit integer
        src = inspect.getsource(func).encode()
        func_id = numpy.array([zlib.adler32(src), zlib.adler32(src[::-1])],
                              dtype=numpy.uint32).view(numpy.int64)[0]

        prototype = make_prototype(func)
        udf = UDF(prototype, func, func_id)
        key = 'pyudf_'+prototype[1]
        storage = remotedict.Storage(key, host=self.host, port=self.port)
        storage[func_id] = udf
        return udf


#### CODE GENERATION


ExtensionFunctionsPython_hpp_template = '''\
/*
  This file is generated using mapd_udf.py script. Do not edit!
*/

#ifdef HAVE_PYTHON
#include <map>
#include <iostream>
#include "Python.h"

int64_t get_pyudf_function(const char * storage_key, int64_t func_id) {{
  static std::map<int64_t,int64_t> cache;
  std::map<int64_t,int64_t>::iterator it=cache.find(func_id);
  if (it != cache.end())
    return it->second;
  int64_t cfunc_ptr = 0;
  PyObject* main_module = PyImport_AddModule("__main__"); // borrowed ref
  PyObject* main_dict = PyModule_GetDict(main_module);    // borrowed ref
  PyObject* storages = PyDict_GetItemString(main_dict, "storages"); // borrowed ref
  if (storages != NULL) {{
    PyObject* pyudf_key = PyUnicode_FromString(storage_key);
    PyObject* storage = PyObject_GetItem(storages, pyudf_key); // this creates remotedict.Storage object
    Py_DECREF(pyudf_key);
    if (storage != NULL) {{
      PyObject* py_func_id = PyLong_FromLongLong(func_id);
      if (py_func_id != NULL) {{
        PyObject* py_udf = PyObject_GetItem(storage, py_func_id); // this connects to remotedict server
        if (py_udf != NULL) {{
          PyObject* py_address = PyObject_GetAttrString(py_udf, "address"); // this calls numba.cfunc and generates machine code
          if (py_address != NULL) {{
            cfunc_ptr = PyLong_AsLongLong(py_address);
            cache[func_id] = cfunc_ptr;
            Py_DECREF(py_address);
          }}
          Py_DECREF(py_udf);
        }}
        Py_DECREF(py_func_id);
      }}
      Py_DECREF(storage);
    }} else {{
      PyErr_SetString(PyExc_KeyError, storage_key);
    }}
  }} else {{
    PyErr_SetString(PyExc_KeyError, "storages is missing in the __main__ namespace");
  }}
  if (PyErr_Occurred()) {{ PyErr_Print(); }}
  return cfunc_ptr;
}}

{pyudfs}

#endif // HAVE_PYTHON
'''

RelAlgTranslator_h_template = '''\
  std::shared_ptr<Analyzer::Expr> translate{Fname}(const RexFunctionOperator*) const;
'''

RelAlgTranslator_cpp_template1 = '''
std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translate{Fname}(
     const RexFunctionOperator* rex_function) const {{
   CHECK_EQ(rex_function->size(), size_t({nargs}+1));
   std::vector<std::shared_ptr<Analyzer::Expr>> args;
   for(int i=0; i<={nargs}; ++i) {{
     const auto arg = translateScalarRex(rex_function->getOperand(i));
     args.push_back(arg);
   }}
   return makeExpr<Analyzer::FunctionOper>(
       rex_function->getType(), rex_function->getName(), args);
}}
'''

RelAlgTranslator_cpp_template2 = '''
if (rex_function->getName() == std::string("{FNAME}")) {{
    return translate{Fname}(rex_function);
}}
'''

pyudf_template = '''
typedef {rtype_t}(*{compact}_typedef)({arg_types});

EXTENSION_NOINLINE
{rtype_t} {Fname}({arg_names_types}) {{
  {compact}_typedef func_ptr = ({compact}_typedef)get_pyudf_function("{fname}", a0);
  //std::cout << "In {Fname}: func_ptr=" << (int64_t)func_ptr << std::endl;
  if (func_ptr == 0) return {nan};
  return (*func_ptr)({arg_names});
}}
'''

MapDSqlOperatorTable_java_template1 = '''\
    opTab.addOperator(new {Fname}());
'''

MapDSqlOperatorTable_java_template2 = '''
public static class {Fname} extends SqlFunction {{
  {Fname}() {{
    super("{FNAME}",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family({SqlTypeFamilies}),
      SqlFunctionCategory.SYSTEM);
  }}
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {{
    assert opBinding.getOperandCount() == ({nargs}+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType({SqlTypeName});
  }}
}}
'''

def generate_pyudf_sources(
        scalar_types=(numba.float64, #numba.float32,
                      numba.int64, #numba.int32, 
                      #numba.boolean,
        ),
        max_nargs=2,
):

    nan_map = dict(int64='-1', int32='-1', int16='-1', int8='-1',
                   uint64='-1', uint32='-1', uint16='-1', uint8='-1',
                   float64='nan("")', float32='nan("")',
                   bool='false',
    )
    for nargs in range(1,max_nargs+1):
        for atypes in itertools.product(scalar_types, repeat=nargs):
            for rtype in scalar_types:
                #print (atypes, rtype)
                rtype_t = c_numba_type(rtype)
                arg_types = ', '.join(['{}'.format(c_numba_type(t)) for i,t in enumerate(atypes)])
                arg_names_types = ', '.join(['const {} a{}'.format(c_numba_type(t), i) for i,t in enumerate((numba.int64,)+atypes)])
                arg_names = ', '.join(['a{}'.format(i+1) for i in range(len(atypes))])           
                compact = compact_prototype(atypes, rtype)
                nan = nan_map[rtype.name]
                #print(pyudf_template.format_map(locals()))
                fname = 'pyudf_'+compact
                Fname = 'Pyudf_'+compact
                FNAME = fname.upper()
                SqlTypeFamilies = ', '.join(map(SqlTypeFamily_numba_type, (numba.int64,)+atypes))
                SqlTypeName = SqlTypeName_numba_type(rtype)
                l = locals()
                d = dict(
                    compact = compact,
                    pyudf = pyudf_template.format_map(l),
                    rat_h = RelAlgTranslator_h_template.format_map(l),
                    rat_cpp1 = RelAlgTranslator_cpp_template1.format_map(l),
                    rat_cpp2 = RelAlgTranslator_cpp_template2.format_map(l),
                    optab_java1 = MapDSqlOperatorTable_java_template1.format_map(l),
                    optab_java2 = MapDSqlOperatorTable_java_template2.format_map(l)
                )
                yield d
                del d

def generate_all_pyudf_sources():
    generated = set()
    for scalar_types, max_nargs in [
            ((numba.float64,), 10),
            ((numba.float32,), 10),
            ((numba.int64,), 10),
            ((numba.int32,), 10),
            #((numba.boolean,), 10),
            ((numba.float64, numba.float32), 2),
            ((numba.float64, numba.int64), 2),
            ((numba.float64, numba.int32), 2),
            #((numba.float64, numba.boolean), 2),
            ((numba.int64, numba.float64), 2),
            ((numba.int64, numba.float32), 2),
            ((numba.int64, numba.int32), 2),
            #((numba.int64, numba.boolean), 2),
            ((numba.int32, numba.float64), 2),
            ((numba.int32, numba.float32), 2),
            ((numba.int32, numba.int64), 2),
            #((numba.int32, numba.boolean), 2),
            #((numba.boolean, numba.float64), 2),
            #((numba.boolean, numba.float32), 2),
            #((numba.boolean, numba.int64), 2),
            #((numba.boolean, numba.int32), 2),
    ]:
        for d in generate_pyudf_sources(scalar_types=scalar_types,
                                        max_nargs=max_nargs):
            if d['compact'] not in generated:
                generated.add(d['compact'])
                yield d

    print('Total number of generated functions:', len(generated))
    
    
def apply_to_sources():
    pyudfs = []
    rat_hs = []
    rat_cpp1s = []
    rat_cpp2s = []
    optab_java1s = []
    optab_java2s = []
    for d in generate_all_pyudf_sources():
        pyudfs.append(d['pyudf'])
        rat_hs.append(d['rat_h'])
        rat_cpp1s.append(d['rat_cpp1'])
        rat_cpp2s.append(d['rat_cpp2'])
        optab_java1s.append(d['optab_java1'])
        optab_java2s.append(d['optab_java2'])
    pyudfs = ''.join(pyudfs)
    rat_hs = ''.join(rat_hs)
    rat_cpp1s = ''.join(rat_cpp1s)
    rat_cpp2s = ''.join(rat_cpp2s)
    optab_java1s = ''.join(optab_java1s)
    optab_java2s = ''.join(optab_java2s)
    ExtensionFunctionsPython_hpp = ExtensionFunctionsPython_hpp_template.format_map(locals())

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    paths = dict(
        optab_java = os.path.join(
            root_dir,
            'java/calcite/src/main/java/com/mapd/calcite/parser/MapDSqlOperatorTable.java'),
        ExtensionFunctionsPython_hpp = os.path.join(
            root_dir,
            'QueryEngine/ExtensionFunctionsPython.hpp'),
        rat_h = os.path.join(
            root_dir,
            'QueryEngine/RelAlgTranslator.h'),
        rat_cpp = os.path.join(
            root_dir,
            'QueryEngine/RelAlgTranslator.cpp')
    )
    pat1=r'////MAPD_UDF1START-DONOTCHANGETHISCOMMENT.*////MAPD_UDF1END-DONOTCHANGETHISCOMMENT'
    pat2=r'////MAPD_UDF2START-DONOTCHANGETHISCOMMENT.*////MAPD_UDF2END-DONOTCHANGETHISCOMMENT'

    rat_h_repl1='''\
////MAPD_UDF1START-DONOTCHANGETHISCOMMENT
  /* Code within this MAPD_UDF1 block is generated by mapd_udf.py script */
'''+rat_hs+'''\
  ////MAPD_UDF1END-DONOTCHANGETHISCOMMENT'''

    rat_cpp_repl1='''\
////MAPD_UDF1START-DONOTCHANGETHISCOMMENT
  /* Code within this MAPD_UDF1 block is generated by mapd_udf.py script */
'''+rat_cpp1s+'''\
  ////MAPD_UDF1END-DONOTCHANGETHISCOMMENT'''

    rat_cpp_repl2='''\
////MAPD_UDF2START-DONOTCHANGETHISCOMMENT
  /* Code within this MAPD_UDF2 block is generated by mapd_udf.py script */
'''+rat_cpp2s+'''\
  ////MAPD_UDF2END-DONOTCHANGETHISCOMMENT'''

    optab_java_repl1='''\
////MAPD_UDF1START-DONOTCHANGETHISCOMMENT
    /* Code within this MAPD_UDF1 block is generated by mapd_udf.py script */
'''+optab_java1s+'''\
    ////MAPD_UDF1END-DONOTCHANGETHISCOMMENT'''

    optab_java_repl2='''\
////MAPD_UDF2START-DONOTCHANGETHISCOMMENT
    /* Code within this MAPD_UDF2 block is generated by mapd_udf.py script */
'''+optab_java2s+'''\
    ////MAPD_UDF2END-DONOTCHANGETHISCOMMENT'''

    ExtensionFunctionsPython_hpp_orig = open(paths['ExtensionFunctionsPython_hpp']).read()
    
    rat_h_src = rat_h_src_orig = open(paths['rat_h']).read()
    rat_h_src = re.sub(pat1, rat_h_repl1, rat_h_src,
                       flags=re.MULTILINE|re.DOTALL)

    rat_cpp_src = rat_cpp_src_orig = open(paths['rat_cpp']).read()
    rat_cpp_src = re.sub(pat1, rat_cpp_repl1, rat_cpp_src,
                         flags=re.MULTILINE|re.DOTALL)
    rat_cpp_src = re.sub(pat2, rat_cpp_repl2, rat_cpp_src,
                         flags=re.MULTILINE|re.DOTALL)

    optab_java_src = optab_java_src_orig = open(paths['optab_java']).read()
    optab_java_src = re.sub(pat1, optab_java_repl1, optab_java_src,
                            flags=re.MULTILINE|re.DOTALL)
    optab_java_src = re.sub(pat2, optab_java_repl2, optab_java_src,
                            flags=re.MULTILINE|re.DOTALL)

    if ExtensionFunctionsPython_hpp != ExtensionFunctionsPython_hpp_orig:
        print('Writing', paths['ExtensionFunctionsPython_hpp'])
        f = open(paths['ExtensionFunctionsPython_hpp'],'w')
        f.write(ExtensionFunctionsPython_hpp)
        f.close()

    if rat_h_src != rat_h_src_orig:
        print('Writing', paths['rat_h'])
        f = open(paths['rat_h'],'w')
        f.write(rat_h_src)
        f.close()

    if rat_cpp_src != rat_cpp_src_orig:
        print('Writing', paths['rat_cpp'])
        f = open(paths['rat_cpp'],'w')
        f.write(rat_cpp_src)
        f.close()

    if optab_java_src != optab_java_src_orig:
        print('Writing', paths['optab_java'])
        f = open(paths['optab_java'],'w')
        f.write(optab_java_src)
        f.close()

if __name__ == '__main__':
    apply_to_sources()
