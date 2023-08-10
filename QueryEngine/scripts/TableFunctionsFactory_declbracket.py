import TableFunctionsFactory_util as util


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
        return '%s<%s>' % (self.name, ', '.join(map(util.tostring, self.args)))

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
        return self.name.rsplit("::", 1)[-1] in util.OutputBufferSizeTypes

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
        elif name in ['GeoPoint', 'GeoLineString', 'GeoPolygon',
                      'GeoMultiPoint', 'GeoMultiLineString', 'GeoMultiPolygon']:
            ctype = name
        else:
            raise NotImplementedError(self)
        if clsname is None:
            return ctype
        if subclsname is None:
            return '%s<%s>' % (clsname, ctype)
        return '%s<%s<%s>>' % (clsname, subclsname, ctype)

    def format_cpp_type(self, idx, use_generic_arg_name=False, real_arg_name=None, is_input=True):
        # Arguments that types are derived from arithmetic and bool
        # types, should be passed by value:
        pass_by_value_typs =(
            'bool', 'int8_t', 'int16_t', 'int32_t', 'int64_t', 'float',
            'double', 'Timestamp', 'DayTimeInterval', 'YearMonthTimeInterval', 'TextEncodingDict')
        # Arguments that types are of struct type, should be passed by
        # reference:
        pass_by_reference_typs =('Column', 'ColumnList', 'TextEncodingNone', 'Array')

        if use_generic_arg_name:
            arg_name = 'input' + str(idx) if is_input else 'output' + str(idx)
        elif real_arg_name is not None:
            arg_name = real_arg_name
        else:
            # in some cases, the real arg name is not specified
            arg_name = 'input' + str(idx) if is_input else 'output' + str(idx)
        const = 'const ' if is_input else ''
        cpp_type = self.get_cpp_type()
        if any(cpp_type.startswith(t) for t in pass_by_reference_typs):
            return '%s%s& %s' % (const, cpp_type, arg_name), arg_name
        elif any(cpp_type.startswith(t) for t in pass_by_value_typs):
            return '%s %s' % (cpp_type, arg_name), arg_name
        else:
            msg =('Argument passing policy for argument `%s` of C++ type `%s` is unspecified.'
                  ' Update pass_by_value_typs or pass_by_reference_typs in'
                  ' Declaration.format_cpp_type to indicate the desired argument'
                  ' passing policy.') %(arg_name, cpp_type)
            raise NotImplementedError(msg)

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
                i = util.find_comma(rest)
                if i == -1:
                    a, rest = rest, ''
                else:
                    a, rest = rest[:i].rstrip(), rest[i + 1:].lstrip()
                args.append(cls.parse(a))
            args = tuple(args)

        name = util.translate_map.get(name, name)
        return cls(name, args)
