__all__ = ['TransformerException', 'AstPrinter', 'TemplateTransformer',
           'FixRowMultiplierPosArgTransformer', 'RenameNodesTransformer',
           'TextEncodingDictTransformer', 'FieldAnnotationTransformer',
           'SupportedAnnotationsTransformer', 'RangeAnnotationTransformer',
           'CursorAnnotationTransformer', 'AmbiguousSignatureCheckTransformer',
           'DefaultValueAnnotationTransformer',
           'DeclBracketTransformer', 'Pipeline']


import sys
import copy
import warnings
import itertools
from ast import literal_eval
from abc import abstractmethod

if sys.version_info > (3, 0):
    from abc import ABC
else:
    from abc import ABCMeta as ABC


import TableFunctionsFactory_util as util
import TableFunctionsFactory_node as tf_node
import TableFunctionsFactory_declbracket as declbracket


class TransformerException(Exception):
    pass


class TransformerWarning(UserWarning):
    pass


class AstVisitor(object):
    __metaclass__ = ABC

    @abstractmethod
    def visit_udtf_node(self, node):
        raise NotImplementedError()

    @abstractmethod
    def visit_composed_node(self, node):
        raise NotImplementedError()

    @abstractmethod
    def visit_arg_node(self, node):
        raise NotImplementedError()

    @abstractmethod
    def visit_primitive_node(self, node):
        raise NotImplementedError()

    @abstractmethod
    def visit_annotation_node(self, node):
        raise NotImplementedError()

    @abstractmethod
    def visit_template_node(self, node):
        raise NotImplementedError()


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
            return util.translate_map.get(composed_node.type) + "<%s>" % (N,)
        if composed_node.is_cursor():
            # Cursor<T1, T2, ..., TN>
            Ts = ", ".join([i.accept(self) for i in composed_node.inner])
            return "Cursor<%s>" % (Ts)
        raise ValueError(composed_node)

    def visit_primitive_node(self, primitive_node):
        t = primitive_node.type
        if primitive_node.is_output_buffer_sizer():
            # arg_pos is zero-based
            return util.translate_map.get(t, t) + "<%d>" % (
                primitive_node.get_parent(tf_node.ArgNode).arg_pos + 1,
            )
        return util.translate_map.get(t, t)


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
            udtf = tf_node.UdtfNode(name, inputs, outputs, udtf_node.annotations, None, udtf_node.sizer, udtf_node.line)
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
            pos = tf_node.PrimitiveNode(str(primitive_node.get_parent(tf_node.ArgNode).arg_pos + 1))
            node = tf_node.ComposedNode(t, inner=[pos])
            return node

        return primitive_node


class RenameNodesTransformer(AstTransformer):
    def visit_primitive_node(self, primitive_node):
        """
        * Rename nodes using util.translate_map as dictionary
            int -> Int32
            float -> Float
        """
        t = primitive_node.type
        return primitive_node.copy(util.translate_map.get(t, t))


class TextEncodingDictTransformer(AstTransformer):
    def visit_udtf_node(self, udtf_node):
        """
        * Add default_input_id to Column(List)<TextEncodingDict> without one
        """
        udtf_node = super(type(self), self).visit_udtf_node(udtf_node)
        # add default input_id
        default_input_id = None
        for idx, t in enumerate(udtf_node.inputs):

            if not isinstance(t.type, tf_node.ComposedNode):
                continue
            if default_input_id is not None:
                pass
            elif t.type.is_column_text_encoding_dict() or t.type.is_column_array_text_encoding_dict():
                default_input_id = tf_node.AnnotationNode('input_id', 'args<%s>' % (idx,))
            elif t.type.is_column_list_text_encoding_dict():
                default_input_id = tf_node.AnnotationNode('input_id', 'args<%s, 0>' % (idx,))

        for t in udtf_node.outputs:
            if isinstance(t.type, tf_node.ComposedNode) and t.type.is_any_text_encoding_dict():
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

            if not isinstance(t.type, tf_node.ComposedNode):
                continue

            if t.type.is_cursor() and t.get_annotation('fields') is None:
                fields = list(tf_node.PrimitiveNode(a.get_annotation('name', 'field%s' % i)) for i, a in enumerate(t.type.inner))
                t.annotations.append(tf_node.AnnotationNode('fields', fields))

        return udtf_node


class DefaultValueAnnotationTransformer(AstTransformer):
    def visit_udtf_node(self, udtf_node):
        """
        * Typechecks default value annotations.
        """
        udtf_node = super(type(self), self).visit_udtf_node(udtf_node)

        for t in udtf_node.inputs:
            for a in filter(lambda x: x.key == "default", t.annotations):
                if not t.type.is_scalar():
                    raise TransformerException(
                        'Error in function "%s", input annotation \'%s=%s\'. '
                        '\"default\" annotation is only supported for scalar types!'\
                        % (udtf_node.name, a.key, a.value)
                    )
                literal = literal_eval(a.value)
                lst = [(bool, 'is_boolean_scalar'), (int, 'is_integer_scalar'), (float, 'is_float_scalar'),
                (str, 'is_string_scalar')]

                for (cls, mthd) in lst:
                    if type(literal) is cls:
                        assert isinstance(t, tf_node.ArgNode)
                        m = getattr(t.type, mthd)
                        if not m():
                            raise TransformerException(
                                'Error in function "%s", input annotation \'%s=%s\'. '
                                'Argument is of type "%s" but value type was inferred as "%s".'
                                % (udtf_node.name, a.key, a.value, t.type.type, type(literal).__name__))
                        break

        return udtf_node


class SupportedAnnotationsTransformer(AstTransformer):
    """
    * Checks for supported annotations in a UDTF
    """
    def visit_udtf_node(self, udtf_node):
        for t in udtf_node.inputs:
            for a in t.annotations:
                if a.key not in util.SupportedAnnotations:
                    raise TransformerException('unknown input annotation: `%s`' % (a.key))
        for t in udtf_node.outputs:
            for a in t.annotations:
                if a.key not in util.SupportedAnnotations:
                    raise TransformerException('unknown output annotation: `%s`' % (a.key))
        for annot in udtf_node.annotations:
            if annot.key not in util.SupportedFunctionAnnotations:
                raise TransformerException('unknown function annotation: `%s`' % (annot.key))
            if annot.value.lower() in ['enable', 'on', '1', 'true']:
                annot.value = '1'
            elif annot.value.lower() in ['disable', 'off', '0', 'false']:
                annot.value = '0'
        return udtf_node


class AmbiguousSignatureCheckTransformer(AstTransformer):
    """
    * A UDTF declaration is ambiguous if two or more ColumnLists are adjacent
    to each other:
        func__0(ColumnList<T> X, ColumnList<T> Z) -> Column<U>
        func__1(ColumnList<T> X, Column<T> Y, ColumnList<T> Z) -> Column<U>
    The first ColumnList ends up consuming all of the arguments leaving a single
    one for the last ColumnList. In other words, Z becomes a Column
    """
    def check_ambiguity(self, udtf_name, lst):
        """
        udtf_name: str
        lst: list[list[Node]]
        """
        for l in lst:
            for i in range(len(l)):
                if not l[i].is_column_list():
                    i += 1
                    continue

                collist = l[i]
                T = collist.inner[0]

                for j in range(i+1, len(l)):
                    # if lst[j] == Column<T>, just continue
                    if l[j].is_column() and l[j].is_column_of(T):
                        continue
                    elif l[j].is_column_list() and l[j].is_column_list_of(T):
                        msg = ('%s signature is ambiguous as there are two '
                            'ColumnList with the same subtype in the same '
                            'group.') % (udtf_name)
                        if udtf_name not in ['ct_overload_column_list_test2__cpu_template']:
                            # warn only when the function ought to be fixed
                            warnings.warn(msg, TransformerWarning)
                    else:
                        break

    def visit_udtf_node(self, udtf_node):
        lst = []
        cursor = False
        for arg in udtf_node.inputs:
            s = arg.accept(self)
            if isinstance(s, list):
                lst.append(s)  # Cursor
                cursor = True
            else:
                # Aggregate single arguments in a list
                if cursor or len(lst) == 0:
                    lst.append([s])
                else:
                    lst[-1].append(s)
                cursor = False

        self.check_ambiguity(udtf_node.name, lst)

        return udtf_node

    def visit_composed_node(self, composed_node):
        s = super(type(self), self).visit_composed_node(composed_node)
        if composed_node.is_cursor():
            return [i.accept(self) for i in composed_node.inner]
        return s

    def visit_arg_node(self, arg_node):
        # skip annotations
        return arg_node.type.accept(self)


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

                v = ann.value
                if len(v) == 2:
                    lo, hi = ann.value
                    value = '"{lo} <= {name} && {name} <= {hi}"'.format(lo=lo, hi=hi, name=name)
                else:
                    raise TransformerException('"range" requires an interval. Got {v}'.format(v=v))
                arg_node.set_annotation('require', value)
        return arg_node


class CursorAnnotationTransformer(AstTransformer):
    """
    * Move a "require" annotation from inside a cursor to the cursor
    """

    def visit_arg_node(self, arg_node):
        if arg_node.type.is_cursor():
            for inner in arg_node.type.inner:
                for ann in inner.annotations:
                    if ann.key == 'require':
                        arg_node.annotations.append(ann)
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

        return util.Signature(name, inputs, outputs, input_annotations, output_annotations, function_annotations, sizer)

    def visit_arg_node(self, arg_node):
        t = arg_node.type.accept(self)
        anns = [a.accept(self) for a in arg_node.annotations]
        return declbracket.Declaration(t, anns)

    def visit_composed_node(self, composed_node):
        typ = util.translate_map.get(composed_node.type, composed_node.type)
        inner = [i.accept(self) for i in composed_node.inner]
        if composed_node.is_cursor():
            inner = list(map(lambda x: x.apply_column(), inner))
            return declbracket.Bracket(typ, args=tuple(inner))
        elif composed_node.is_output_buffer_sizer():
            return declbracket.Bracket(typ, args=tuple(inner))
        else:
            return declbracket.Bracket(typ + str(inner[0]))

    def visit_primitive_node(self, primitive_node):
        t = primitive_node.type
        return declbracket.Bracket(t)

    def visit_annotation_node(self, annotation_node):
        key = annotation_node.key
        value = annotation_node.value
        return (key, value)


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
