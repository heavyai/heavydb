#cython: language_level=3

from cython.operator cimport dereference as deref
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libc.stdint cimport int64_t, uint64_t, uint32_t
from libcpp.map cimport map
from libcpp.memory cimport shared_ptr
from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from datetime import datetime
from collections import namedtuple

import os
import sys

from DBEngine cimport *
from DBEngine cimport ColumnType as _ColumnType
from DBEngine cimport ColumnEncoding as _ColumnEncoding
from DBEngine cimport ColumnDetails as _ColumnDetails
from DBEngine cimport Row as _Row
from DBEngine cimport Cursor as _Cursor
from DBEngine cimport DBEngine
from pyarrow.lib cimport *

cdef class PyColumnType:
  cdef _ColumnType c_column_type

  def __cinit__(self, int val):
    self.c_column_type = <_ColumnType> val

  def __eq__(self, val):
    if isinstance(val, int):
        return val == <int> self.c_column_type
    return False

  def to_str(self):
    cdef c = {
        <int>SMALLINT : "SMALLINT",
        <int>INT : "INT",
        <int>BIGINT : "BIGINT",
        <int>FLOAT : "FLOAT",
        <int>DECIMAL : "DECIMAL",
        <int>DOUBLE : "DOUBLE",
        <int>STR : "STR",
        <int>TIME : "TIME",
        <int>TIMESTAMP : "TIMESTAMP",
        <int>DATE : "DATE",
        <int>BOOL : "BOOL",
        <int>INTERVAL_DAY_TIME : "INTERVAL_DAY_TIME",
        <int>INTERVAL_YEAR_MONTH : "INTERVAL_YEAR_MONTH",
        <int>POINT : "POINT",
        <int>LINESTRING : "LINESTRING",
        <int>POLYGON : "POLYGON",
        <int>MULTIPOLYGON : "MULTIPOLYGON",
        <int>TINYINT : "TINYINT",
        <int>GEOMETRY : "GEOMETRY",
        <int>GEOGRAPHY : "TIME",
        <int>UNKNOWN : "UNKNOWN"}
    return c[<int>self.c_column_type]

cdef class PyColumnEncoding:
  cdef _ColumnEncoding c_column_enc

  def __cinit__(self, int val):
    self.c_column_enc = <_ColumnEncoding> val

  def __eq__(self, val):
    if isinstance(val, int):
        return val == <int> self.c_column_enc
    return False

  def to_str(self):
    cdef c = {
        <int>NONE : "NONE",
        <int>FIXED : "FIXED",
        <int>RL : "RL",
        <int>DIFF : "DIFF",
        <int>DICT : "DICT",
        <int>SPARSE : "SPARSE",
        <int>GEOINT : "GEOINT",
        <int>DATE_IN_DAYS : "DATE_IN_DAYS"}
    return c[<int>self.c_column_enc]


cdef class PyColumnDetails:
    cdef _ColumnDetails c_col
    def __cinit__(self, string col_name, int col_type, int col_enc, bool nullable, bool is_array, int precision, int scale, int comp_param):
        self.c_col = _ColumnDetails(col_name, <_ColumnType>col_type, <_ColumnEncoding>col_enc, nullable, is_array, precision, scale, comp_param)


cdef class PyRow:
    cdef _Row c_row  #Hold a C++ instance which we're wrapping

    def getField(self, col_num, col_type):
        if col_type == <int>INT:
            return self.c_row.getInt(col_num);
        if col_type == <int>FLOAT:
            return self.c_row.getFloat(col_num);
        if col_type == <int>DOUBLE:
            return self.c_row.getDouble(col_num);
        if col_type == <int>STR:
            return self.c_row.getStr(col_num);
        return "Unknown type"


cdef class PyCursor:
    cdef shared_ptr[_Cursor] c_cursor  #Hold a C++ instance which we're wrapping
    cdef shared_ptr[CRecordBatch] c_batch
    cdef shared_ptr[CTable] c_table

    def colCount(self):
        return self.c_cursor.get().getColCount()

    def rowCount(self):
        return self.c_cursor.get().getRowCount()

    def nextRow(self):
        obj = PyRow()
        obj.c_row = self.c_cursor.get().getNextRow()
        return obj

    def getColType(self, uint32_t pos):
        obj = PyColumnType(<int>self.c_cursor.get().getColType(pos))
        return obj

    def showRows(self, int max_rows=0):
        col_count = self.colCount();
        row_count = self.rowCount();
        if max_rows > 0 and row_count > max_rows:
            row_count = max_rows
        col_types = [];
        col_types_str = [];
        for i in range(col_count):
            ct = self.getColType(i)
            col_types.append(ct)
            col_types_str.append(ct.to_str())
        format_row = "{:>12}" * (len(col_types) + 1)
        print(*col_types_str)
        for j in range(row_count):
            r = self.nextRow()
            fields = []
            for f in range(col_count):
                fields.append(r.getField(f, col_types[f]))
            print(*fields, flush=True)

    def getArrowRecordBatch(self):
        with nogil:
            self.c_batch = self.c_cursor.get().getArrowRecordBatch()
        if self.c_batch.get() is NULL:
            return None
        else:
            prb = pyarrow_wrap_batch(self.c_batch)
            return prb

    def getArrowTable(self):
        with nogil:
            self.c_table = self.c_cursor.get().getArrowTable()
        if self.c_table.get() is NULL:
            return None
        else:
            return pyarrow_wrap_table(self.c_table)

ColumnDetailsTp = namedtuple("ColumnDetails", ["name", "type", "nullable",
                                             "precision", "scale",
                                             "comp_param", "encoding",
                                             "is_array"])
cdef class PyDbEngine:
    cdef shared_ptr[DBEngine] c_dbe  #Hold a C++ instance which we're wrapping
    cdef map[string, string] c_parameters

    def __cinit__(self, **kwargs):
        cmd_str = "".join(' --%s %r' % x for x in kwargs.iteritems())
        cmd_str = cmd_str.replace("_", "-")
        self.c_dbe = DBEngine.create(bytes(cmd_str, 'utf-8'))
        if self.closed:
            raise RuntimeError('Initialization failed')

    @property
    def closed(self):
        return self.c_dbe == NULL

    def close(self):
        pass

    def check_closed(self):
        if self.closed:
            raise RuntimeError('DB engine uninitialized')

    def executeDDL(self, query):
        self.check_closed()
        self.c_dbe.get().executeDDL(bytes(query, 'utf-8'))

    def executeDML(self, query):
        self.check_closed()
        obj = PyCursor();
        obj.c_cursor = self.c_dbe.get().executeDML(bytes(query, 'utf-8'));
        return obj;

    def executeRA(self, query):
        self.check_closed()
        obj = PyCursor();
        obj.c_cursor = self.c_dbe.get().executeRA(bytes(query, 'utf-8'));
        return obj

    def importArrowTable(self, name, table, **kwargs):
        self.check_closed()
        cdef shared_ptr[CTable] t = pyarrow_unwrap_table(table)
        cdef string n = bytes(name, 'utf-8')
        cdef uint64_t fragment_size = kwargs.get("fragment_size", 0)
        if n.empty() or not t.get():
            raise RuntimeError('Table initialization failed')
        self.check_closed()
        self.c_dbe.get().importArrowTable(n, t, fragment_size)

    # TODO: remove this legacy alias.
    def consumeArrowTable(self, name, table, **kwargs):
        return self.importArrowTable(name, table, **kwargs)

    def select_df(self, query):
        obj = self.executeDML(query)
        if not obj:
            raise RuntimeError("Cursor uninitialized")
        prb = obj.getArrowRecordBatch()
        return prb.to_pandas()

    def get_table_details(self, table_name):
        self.check_closed()
        cdef vector[ColumnDetails] table_details = self.c_dbe.get().getTableDetails(bytes(table_name, 'utf-8'))
        return [
            ColumnDetailsTp(x.col_name, PyColumnType(<int>x.col_type).to_str(),
                            x.nullable, x.precision, x.scale, x.comp_param,
                            PyColumnEncoding(<int>x.encoding).to_str(), x.is_array)
            for x in table_details
        ]

    def get_tables(self):
        self.check_closed()
        return self.c_dbe.get().getTables()

    def setDatabase(self, db_name):
        self.check_closed()
        return self.c_dbe.get().setDatabase(db_name)

    def login(self, db_name, user_name, password):
        self.check_closed()
        return self.c_dbe.get().login(db_name, user_name, password)

