from __future__ import absolute_import

from libc.stdint cimport int64_t, uint64_t, uint32_t, uint8_t
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp cimport bool, nullptr_t, nullptr
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref

cdef extern from "arrow/api.h" namespace "arrow" nogil:
    cdef cppclass CRecordBatch" arrow::RecordBatch":
        int num_columns()
        int64_t num_rows()

cdef extern from "boost/variant.hpp" namespace "boost":
    cdef cppclass boostvariant "boost::variant" [T1]:
        pass
    cdef cppclass boostvariant2 "boost::variant" [T1, T2]:
        pass
    cdef cppclass boostvariant4 "boost::variant" [T1, T2, T3, T4]:
        pass

ctypedef boostvariant2[string, void*] NullableString

ctypedef boostvariant4[int64_t, double, float, NullableString] ScalarTargetValue

cdef extern from "boost/optional.hpp" namespace "boost":
    cdef cppclass boostoptional "boost::optional" [T]:
        pass

ctypedef boostoptional[vector[ScalarTargetValue]] boost_optional_vector

ctypedef boostvariant[boost_optional_vector] ArrayTargetValue

ctypedef boostvariant2[ScalarTargetValue, ArrayTargetValue] TargetValue

cdef extern from "QueryEngine/TargetValue.h":
    cdef cppclass ResultSet:
        ResultSet()
        string getName() except *
        vector[TargetValue] getNextRow(const bool translate_strings, const bool decimal_to_double)

cdef extern from "QueryEngine/ResultSet.h":
    cdef cppclass ResultSet:
        size_t colCount()
        size_t rowCount(const bool force_parallel)
        vector[TargetValue] getNextRow(const bool translate_strings, const bool decimal_to_double)
        TargetValue getRowAt(const size_t row_idx, const size_t col_idx, const bool translate_strings, const bool decimal_to_double)

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
        int64_t getInt(size_t col)
        double getDouble(size_t col)
        string getStr(size_t col)

cdef extern from "DBEngine.h" namespace 'EmbeddedDatabase':
    cdef cppclass Cursor:
        size_t getColCount()
        size_t getRowCount()
        Row getNextRow()
        ColumnType getColType(uint32_t nPos)
        shared_ptr[CRecordBatch] getArrowRecordBatch() nogil except +

    cdef cppclass DBEngine:
        void executeDDL(string)
        Cursor* executeDML(string)
#        shared_ptr[ResultSet] executeDML(string)
        vector[string] getTables()
        vector[ColumnDetails] getTableDetails(string)
        void reset()
        @staticmethod
        DBEngine* create(string, int, bool)

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
