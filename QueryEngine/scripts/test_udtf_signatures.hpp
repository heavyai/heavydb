/*
  This file contains test inputs to generate_TableFunctionsFactory_init.py script.
*/

// clang-format off
/*

  UDTF: foo_0(Column<int32>) -> Column<int32> $=>$
        foo_0(ColumnInt32) -> ColumnInt32

  UDTF: foo_1(Column<int32>, RowMultiplier) -> Column<int32> $=>$
        foo_1(ColumnInt32, kUserSpecifiedRowMultiplier<2>) -> ColumnInt32

  UDTF: foo_2(RowMultiplier, Column<int32>) -> Column<int32> $=>$
        foo_2(kUserSpecifiedRowMultiplier<1>, ColumnInt32) -> ColumnInt32

  UDTF: foo_3(Column<int32>, Constant<5>) -> Column<int32> $=>$
        foo_3(ColumnInt32, kConstant<5>) -> ColumnInt32

  UDTF: foo_4(Column<int32>, int i, int j) -> Column<int32> | output_row_size="i * j" $=>$
        foo_4(ColumnInt32, Int32 | name=i, Int32 | name=j) -> ColumnInt32 | kPreFlightParameter="i * j"

  UDTF: foo_5(Column<int32>, int i, int j) -> Column<int32> | output_row_size=5 $=>$
        foo_5(ColumnInt32, Int32 | name=i, Int32 | name=j) -> ColumnInt32 | kPreFlightParameter=5

  UDTF: foo_6(Column<int32>, int i, int j) -> Column<int32> | output_row_size=i $=>$
        foo_6(ColumnInt32, Int32 | name=i, Int32 | name=j) -> ColumnInt32 | kPreFlightParameter=i

  UDTF: foo_7(Cursor<Column<int32>>) -> Column<int32> $=>$
        foo_7(Cursor<ColumnInt32> | fields=[field0]) -> ColumnInt32

  UDTF: foo_8(Cursor<Column<int32>, Column<float>>) -> Column<int32> $=>$
        foo_8(Cursor<ColumnInt32, ColumnFloat> | fields=[field0,field1]) -> ColumnInt32
  UDTF: foo_9(Cursor<Column<int32>>, Cursor<Column<float>>) -> Column<int32> $=>$
        foo_9(Cursor<ColumnInt32> | fields=[field0], Cursor<ColumnFloat> | fields=[field0]) -> ColumnInt32

  UDTF: foo_10(Column<int32>) -> Column<int32>, Column<float> $=>$
        foo_10(ColumnInt32) -> ColumnInt32, ColumnFloat
  UDTF: foo_11(Column<int32>|name=a) -> Column<int32>|name=out $=>$
        foo_11(ColumnInt32 | name=a) -> ColumnInt32 | name=out
  UDTF: foo_12(Column<int32>a) -> Column<int32>out $=>$
        foo_12(ColumnInt32 | name=a) -> ColumnInt32 | name=out

  UDTF: foo_13(Column<TextEncodingDict>) -> Column<TextEncodingDict> $=>$
        foo_13(ColumnTextEncodingDict) -> ColumnTextEncodingDict | input_id=args<0>
  UDTF: foo_14(ColumnList<TextEncodingDict>) -> Column<TextEncodingDict> $=>$
        foo_14(ColumnListTextEncodingDict) -> ColumnTextEncodingDict | input_id=args<0, 0>
  UDTF: foo_15(ColumnList<TextEncodingDict>, Column<int>, Column<TextEncodingDict>) -> Column<TextEncodingDict> | input_id=args<2> $=>$
        foo_15(ColumnListTextEncodingDict, ColumnInt32, ColumnTextEncodingDict) -> ColumnTextEncodingDict | input_id=args<2>
  UDTF: foo_16(Column<int>, Column<TextEncodingDict>) -> Column<TextEncodingDict> | input_id=args<1> $=>$
        foo_16(ColumnInt32, ColumnTextEncodingDict) -> ColumnTextEncodingDict | input_id=args<1>
  UDTF: foo_17(Column<int>, ColumnList<TextEncodingDict>) -> Column<TextEncodingDict> $=>$
        foo_17(ColumnInt32, ColumnListTextEncodingDict) -> ColumnTextEncodingDict | input_id=args<1, 0>
  UDTF: foo_18(Column<TextEncodingDict> | name = a) -> Column<TextEncodingDict> | name=out | input_id=args<0> $=>$
        foo_18(ColumnTextEncodingDict | name=a) -> ColumnTextEncodingDict | name=out | input_id=args<0>
  UDTF: foo_19(Column<TextEncodingDict> a) -> Column<TextEncodingDict> out | input_id=args<0> $=>$
        foo_19(ColumnTextEncodingDict | name=a) -> ColumnTextEncodingDict | name=out | input_id=args<0>

  UDTF: foo_20__cpu_template(Column<int32_t>) -> Column<int32_t> $=>$
        foo_20__cpu_template(ColumnInt32) -> ColumnInt32

  UDTF: foo_21__cpu(Column<T>, T, Cursor<ColumnList<U>>) -> Column<T>, T=[int32], U=[float] $=>$
        foo_21__cpu(ColumnInt32, Int32, Cursor<ColumnListFloat> | fields=[field0]) -> ColumnInt32
  UDTF: foo_22__cpu(Column<T> in1, T in2, Cursor<ColumnList<U>> in3) -> Column<T> out1, Column<U> out2, T=[int32], U=[float] $=>$
        foo_22__cpu(ColumnInt32 | name=in1, Int32 | name=in2, Cursor<ColumnListFloat> | name=in3 | fields=[field0]) -> ColumnInt32 | name=out1, ColumnFloat | name=out2

  UDTF: foo_23__cpu_template(Column<T>) -> Column<U>, T=[int32, int64], U=[float, double] $=>$
        foo_23__cpu_template(ColumnInt32) -> ColumnFloat  $=>$
        foo_23__cpu_template(ColumnInt64) -> ColumnFloat  $=>$
        foo_23__cpu_template(ColumnInt32) -> ColumnDouble $=>$
        foo_23__cpu_template(ColumnInt64) -> ColumnDouble

  UDTF: foo_24__cpu(TableFunctionManager, int64_t) -> Column<int64_t> $=>$
        foo_24__cpu(TableFunctionManager, Int64) -> ColumnInt64

  UDTF: foo_25__cpu(TableFunctionManager, Cursor<Column<int64_t> x, Column<int64_t> y> z) -> Column<int64_t> $=>$
        foo_25__cpu(TableFunctionManager, Cursor<ColumnInt64 | name=x, ColumnInt64 | name=y> | name=z | fields=[x,y]) -> ColumnInt64

  UDTF: foo_26__cpu(TableFunctionManager) | filter_table_function_transpose=on -> Column<int64_t> $=>$
        foo_26__cpu(TableFunctionManager) | filter_table_function_transpose=1 -> ColumnInt64
  UDTF: foo_27__cpu(TableFunctionManager) | filter_table_function_transpose=off -> Column<int64_t> $=>$
        foo_27__cpu(TableFunctionManager) | filter_table_function_transpose=0 -> ColumnInt64
  UDTF: foo_28__cpu(TableFunctionManager) | bar=off -> Column<int64_t> $=>$
        TransformerException: unknown function annotation: `bar`
  UDTF: foo_29__cpu(TableFunctionManager | bar=off) -> Column<int64_t> $=>$
        TransformerException: unknown input annotation: `bar`
  UDTF: foo_30__cpu(TableFunctionManager) -> Column<int64_t> | bar=off $=>$
        TransformerException: unknown output annotation: `bar`

  UDTF: foo_31__cpu(TableFunctionManager, Cursor<int32_t x> | fields=[x1]) -> Column<int64_t>!
        foo_31__cpu(TableFunctionManager, Cursor<Int32 | name=x> | fields=[x1]) -> ColumnInt64

  UDTF: foo_32_int_require(Column<int> | name=col, int32_t | require="sqrt(arg1) > col.size()" | name=arg1) -> int32_t $=>$
        foo_32_int_require(ColumnInt32 | name=col, Int32 | require="sqrt(arg1) > col.size()" | name=arg1) -> Int32
  UDTF: foo_33_int_require_mgr(TableFunctionManager, Column<int> | name=col, int32_t | require="sqrt(arg1) > col.size()" | name=arg1) -> int32_t $=>$
        foo_33_int_require_mgr(TableFunctionManager, ColumnInt32 | name=col, Int32 | require="sqrt(arg1) > col.size()" | name=arg1) -> Int32
  UDTF: foo_34_int_require(Column<int> col, int x | require="x > 0" | require="x < 5") -> int $=>$
        foo_34_int_require(ColumnInt32 | name=col, Int32 | name=x | require="x > 0" | require="x < 5") -> Int32
  UDTF: foo_35_str_require(TextEncodingNone s | require="s == \"str\"") -> int32_t $=>$
        foo_35_str_require(TextEncodingNone | name=s | require="s == \"str\"") -> Int32
  UDTF: foo_36_str_require(TextEncodingNone s | require="s != \"str\"") -> int32_t $=>$
        foo_36_str_require(TextEncodingNone | name=s | require="s != \"str\"") -> Int32


  UDTF: foo_37(TableFunctionManager, TextEncodingNone agg_type | require="agg_type == \"MAX\"",
            Cursor<Column<K> x, Column<T> y>) -> Column<int>, K=[int], T=[float] $=>$
        foo_37(TableFunctionManager, TextEncodingNone | name=agg_type | require="agg_type == \"MAX\"", Cursor<ColumnInt32 | name=x, ColumnFloat | name=y> | fields=[x,y]) -> ColumnInt32

  UDTF: foo_38(TableFunctionManager,
               Cursor<Column<int32_t> a, ColumnList<float> b>,
               int32_t x,
               int32_t y | require="x > 0" | require="y > 0") ->
               Column<int32_t>, Column<int32_t> $=>$
        foo_38(TableFunctionManager, Cursor<ColumnInt32 | name=a, ColumnListFloat | name=b> | fields=[a,b], Int32 | name=x, Int32 | name=y | require="x > 0" | require="y > 0") -> ColumnInt32, ColumnInt32

  UDTF: foo_39(TableFunctionManager) -> Column<TextEncodingDict> new_dict | input_id=args<> $=>$
        foo_39(TableFunctionManager) -> ColumnTextEncodingDict | name=new_dict | input_id=args<-1>
 */
// clang-format on
