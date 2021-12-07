from __future__ import absolute_import

from libc.stdint cimport int64_t, uint64_t, uint32_t, uint8_t
from libcpp.map cimport map
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp cimport bool, nullptr_t, nullptr
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref
from pyarrow.lib cimport CTable

cdef extern from "arrow/api.h" namespace "arrow" nogil:
    cdef cppclass CRecordBatch" arrow::RecordBatch":
        int num_columns()
        int64_t num_rows()

    cdef cppclass CTable" arrow::Table":
        pass

cdef extern from "DBETypes.h" namespace 'EmbeddedDatabase':
    cdef cppclass ColumnType:
        pass

    cdef cppclass ColumnEncoding:
        pass

    cdef cppclass ColumnDetails:
        string col_name
        ColumnType col_type
        ColumnEncoding encoding
        bool nullable
        bool is_array
        int precision
        int scale
        int comp_param
        ColumnDetails()
        ColumnDetails(string,ColumnType,ColumnEncoding,bool,bool,int,int,int)

    cdef cppclass Row:
        int64_t getInt(size_t col) except +
        float getFloat(size_t col) except +
        double getDouble(size_t col) except +
        string getStr(size_t col) except +

cdef extern from "DBEngine.h" namespace 'EmbeddedDatabase':
    cdef cppclass Cursor:
        size_t getColCount()
        size_t getRowCount()
        Row getNextRow()
        ColumnType getColType(uint32_t nPos)
        shared_ptr[CRecordBatch] getArrowRecordBatch() nogil except +
        shared_ptr[CTable] getArrowTable() nogil except +

    cdef cppclass DBEngine:
        void executeDDL(string) except +
        shared_ptr[Cursor] executeDML(string) except +
        shared_ptr[Cursor] executeRA(string) except +
        vector[string] getTables() except +
        vector[ColumnDetails] getTableDetails(string) except +
        void importArrowTable(string, shared_ptr[CTable]&, uint64_t) except +
        bool setDatabase(string db_name) except +
        bool login(string db_name, string user_name, string password) except +
        @staticmethod
        shared_ptr[DBEngine] create(string cmd_str) except+

cdef extern from "DBETypes.h" namespace 'EmbeddedDatabase::ColumnType':
    cdef ColumnType SMALLINT
    cdef ColumnType INT
    cdef ColumnType BIGINT
    cdef ColumnType FLOAT
    cdef ColumnType DECIMAL
    cdef ColumnType DOUBLE
    cdef ColumnType STR
    cdef ColumnType TIME
    cdef ColumnType TIMESTAMP
    cdef ColumnType DATE
    cdef ColumnType BOOL
    cdef ColumnType INTERVAL_DAY_TIME
    cdef ColumnType INTERVAL_YEAR_MONTH
    cdef ColumnType POINT
    cdef ColumnType LINESTRING
    cdef ColumnType POLYGON
    cdef ColumnType MULTIPOLYGON
    cdef ColumnType TINYINT
    cdef ColumnType GEOMETRY
    cdef ColumnType GEOGRAPHY
    cdef ColumnType UNKNOWN


cdef extern from "DBETypes.h" namespace 'EmbeddedDatabase::ColumnEncoding':
    cdef ColumnEncoding NONE
    cdef ColumnEncoding FIXED
    cdef ColumnEncoding RL
    cdef ColumnEncoding DIFF
    cdef ColumnEncoding DICT
    cdef ColumnEncoding SPARSE
    cdef ColumnEncoding GEOINT
    cdef ColumnEncoding DATE_IN_DAYS
