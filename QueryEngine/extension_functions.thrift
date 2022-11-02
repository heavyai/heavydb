namespace java ai.heavy.thrift.calciteserver
namespace py heavydb.extension_functions

/* See QueryEngine/ExtensionFunctionsWhitelist.h for required values */
enum TExtArgumentType {
  Int8,
  Int16,
  Int32,
  Int64,
  Float,
  Double,
  Void,
  PInt8,
  PInt16,
  PInt32,
  PInt64,
  PFloat,
  PDouble,
  PBool,
  Bool,
  ArrayInt8,
  ArrayInt16,
  ArrayInt32,
  ArrayInt64,
  ArrayFloat,
  ArrayDouble,
  ArrayBool,
  GeoPoint,
  GeoLineString,
  Cursor,
  GeoPolygon,
  GeoMultiPolygon,
  ColumnInt8,
  ColumnInt16,
  ColumnInt32,
  ColumnInt64,
  ColumnFloat,
  ColumnDouble,
  ColumnBool,
  TextEncodingNone,
  TextEncodingDict,
  ColumnListInt8,
  ColumnListInt16,
  ColumnListInt32,
  ColumnListInt64,
  ColumnListFloat,
  ColumnListDouble,
  ColumnListBool,
  ColumnTextEncodingDict,
  ColumnListTextEncodingDict,
  ColumnTimestamp,
  Timestamp,
  ColumnArrayInt8,
  ColumnArrayInt16,
  ColumnArrayInt32,
  ColumnArrayInt64,
  ColumnArrayFloat,
  ColumnArrayDouble,
  ColumnArrayBool,
  ColumnListArrayInt8,
  ColumnListArrayInt16,
  ColumnListArrayInt32,
  ColumnListArrayInt64,
  ColumnListArrayFloat,
  ColumnListArrayDouble,
  ColumnListArrayBool,
  GeoMultiLineString,
  ArrayTextEncodingNone,
  ColumnTextEncodingNone,
  ColumnListTextEncodingNone,
  ColumnArrayTextEncodingNone,
  ColumnListArrayTextEncodingNone,
  ArrayTextEncodingDict,
  ColumnArrayTextEncodingDict,
  ColumnListArrayTextEncodingDict,
  GeoMultiPoint,
  DayTimeInterval,
  YearMonthTimeInterval,
}

/* See QueryEngine/TableFunctions/TableFunctionsFactory.h for required values */
enum TOutputBufferSizeType {
  kConstant,
  kUserSpecifiedConstantParameter,
  kUserSpecifiedRowMultiplier,
  kTableFunctionSpecifiedParameter,
  kPreFlightParameter,
}

struct TUserDefinedFunction {
  1: string name,
  2: list<TExtArgumentType> argTypes,
  3: TExtArgumentType retType,
  4: bool usesManager,
}

struct TUserDefinedTableFunction {
  /* The signature of an UDTF is defined by the SQL extension function signature
     and the LLVM/NVVM IR function signature. The signature of SQL extension function is
       <name>(<input1>, <input2>, ..., <sizer parameter>, <inputN>, ..) -> table(<output1>, <output2>, ...)
     where input can be either cursor or literal type and are collected in sqlArgTypes.

     The signature of a LLVM IR function is
       int32 <name>(<inputArgTypes>, <outputArgTypes>)
     where
       inputArgTypes[sizerArgPos - 1] corresponds to sizer parameter
       inputArgTypes[-2] corresponds to input_row_count_ptr
       inputArgTypes[-1] corresponds to output_row_count_ptr

      Annotations of UDTF input and output arguments is a mapping of annotation label and the
      corresponding value. The length of the annotations list is len(sqlArgTypes) + len(outputArgTypes).

      Supported annotation labels are:
        * input_id: used for TextEncodingDict output column to set its
          dict_id to the dict_id of TextEncodingDict input column as
          specified by the input_id annotation value.
        * name: specify the name of input or output arguments. This can be
          used to define the names of output columns as well as to improve
          the exception messages when binding fails.
   */
  1: string name,
  2: TOutputBufferSizeType sizerType,
  3: i32 sizerArgPos,
  4: list<TExtArgumentType> inputArgTypes,
  5: list<TExtArgumentType> outputArgTypes,
  6: list<TExtArgumentType> sqlArgTypes,
  7: list<map<string, string>> annotations
}

