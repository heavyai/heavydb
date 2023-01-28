__all__ = ['UdtfNode', 'ArgNode', 'PrimitiveNode', 'ComposedNode',
           'AnnotationNode', 'TemplateNode']


import sys
from abc import abstractmethod

import TableFunctionsFactory_transformers as transformers
import TableFunctionsFactory_util as util

if sys.version_info > (3, 0):
    from abc import ABC
    from collections.abc import Iterable
else:
    from abc import ABCMeta as ABC
    from collections import Iterable


class Node(object):

    __metaclass__ = ABC

    @abstractmethod
    def accept(self, visitor):
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


class PrintNode(object):
    def __str__(self):
        return self.accept(transformers.AstPrinter())

    def __repr__(self):
        return str(self)


class IterableNode(Iterable):
    pass


class UdtfNode(Node, IterableNode, PrintNode):

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


class ArgNode(Node, IterableNode, PrintNode):

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
        self.kind = None

    def accept(self, visitor):
        return visitor.visit_arg_node(self)

    def __iter__(self):
        yield self.type
        for a in self.annotations:
            yield a

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
        return util.translate_map.get(t, t) in util.OutputBufferSizeTypes

    def is_text_encoding_dict(self):
        return self.type == 'TextEncodingDict'

    def is_array_text_encoding_dict(self):
        return self.type == 'ArrayTextEncodingDict'

    def is_integer_scalar(self):
        return self.type.lower() in ('int8_t', 'int16_t', 'int32_t', 'int64_t')

    def is_float_scalar(self):
        return self.type.lower() in ('float', 'double')

    def is_boolean_scalar(self):
        return self.type.lower() == 'bool'

    def is_string_scalar(self):
        # we only support 'TextEncodingNone' string scalars atm
        return self.type == "TextEncodingNone"

    def is_scalar(self):
        return self.is_integer_scalar() or self.is_float_scalar() or self.is_boolean_scalar() or self.is_string_scalar()


class PrimitiveNode(TypeNode, PrintNode):

    def __init__(self, type):
        """
        Parameters
        ----------
        type : str
        """
        self.type = type

    def accept(self, visitor):
        return visitor.visit_primitive_node(self)

    def __eq__(self, other):
        if isinstance(other, PrimitiveNode):
            return self.type == other.type
        return False


class ComposedNode(TypeNode, IterableNode, PrintNode):

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

    def is_column_of(self, T):
        return self.is_column() and self.inner[0] == T

    def is_column_list_of(self, T):
        return self.is_column_list() and self.inner[0] == T


class AnnotationNode(Node, PrintNode):

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


class TemplateNode(Node, PrintNode):

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
