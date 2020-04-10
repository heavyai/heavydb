namespace java com.omnisci.thrift.calciteserver
namespace py omnisci.extension_functions

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
  GeoMultiPolygon
}

/* See QueryEngine/TableFunctions/TableFunctionsFactory.h for required values */
enum TOutputBufferSizeType {
  kUserSpecifiedConstantParameter,
  kUserSpecifiedRowMultiplier,
  kConstant
}

struct TUserDefinedFunction {
  1: string name,
  2: list<TExtArgumentType> argTypes
  3: TExtArgumentType retType
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
   */
  1: string name,
  2: TOutputBufferSizeType sizerType,
  3: i32 sizerArgPos,
  4: list<TExtArgumentType> inputArgTypes,
  5: list<TExtArgumentType> outputArgTypes,
  6: list<TExtArgumentType> sqlArgTypes
}

