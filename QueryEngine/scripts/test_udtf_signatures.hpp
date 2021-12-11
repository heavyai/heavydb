/*
  This file contains test inputs to generate_TableFunctionsFactory_init.py script.
*/

// clang-format off
/*

  UDTF: foo(Column<int32>) -> Column<int32> $=>$
        foo(ColumnInt32) -> ColumnInt32

  UDTF: foo(Column<int32>, RowMultiplier) -> Column<int32> $=>$
        foo(ColumnInt32, kUserSpecifiedRowMultiplier<2>) -> ColumnInt32

  UDTF: foo(RowMultiplier, Column<int32>) -> Column<int32> $=>$
        foo(kUserSpecifiedRowMultiplier<1>, ColumnInt32) -> ColumnInt32

  UDTF: foo(Column<int32>, Constant<5>) -> Column<int32> $=>$
        foo(ColumnInt32, kConstant<5>) -> ColumnInt32

  UDTF: foo(Column<int32>, int i, int j) -> Column<int32> | output_row_size="i * j" $=>$
        foo(ColumnInt32, Int32 | name=i, Int32 | name=j) -> ColumnInt32 | kPreFlightParameter="i * j"

  UDTF: foo(Column<int32>, int i, int j) -> Column<int32> | output_row_size=5 $=>$
        foo(ColumnInt32, Int32 | name=i, Int32 | name=j) -> ColumnInt32 | kPreFlightParameter=5

  UDTF: foo(Column<int32>, int i, int j) -> Column<int32> | output_row_size=i $=>$
        foo(ColumnInt32, Int32 | name=i, Int32 | name=j) -> ColumnInt32 | kPreFlightParameter=i

  UDTF: foo(Cursor<Column<int32>>) -> Column<int32> $=>$
        foo(Cursor<ColumnInt32> | fields=[field0]) -> ColumnInt32

  UDTF: foo(Cursor<Column<int32>, Column<float>>) -> Column<int32> $=>$
        foo(Cursor<ColumnInt32, ColumnFloat> | fields=[field0,field1]) -> ColumnInt32
  UDTF: foo(Cursor<Column<int32>>, Cursor<Column<float>>) -> Column<int32> $=>$
        foo(Cursor<ColumnInt32> | fields=[field0], Cursor<ColumnFloat> | fields=[field0]) -> ColumnInt32

  UDTF: foo(Column<int32>) -> Column<int32>, Column<float> $=>$
        foo(ColumnInt32) -> ColumnInt32, ColumnFloat
  UDTF: foo(Column<int32>|name=a) -> Column<int32>|name=out $=>$
        foo(ColumnInt32 | name=a) -> ColumnInt32 | name=out
  UDTF: foo(Column<int32>a) -> Column<int32>out $=>$
        foo(ColumnInt32 | name=a) -> ColumnInt32 | name=out

  UDTF: foo(Column<TextEncodingDict>) -> Column<TextEncodingDict> $=>$
        foo(ColumnTextEncodingDict) -> ColumnTextEncodingDict | input_id=args<0>
  UDTF: foo(ColumnList<TextEncodingDict>) -> Column<TextEncodingDict> $=>$
        foo(ColumnListTextEncodingDict) -> ColumnTextEncodingDict | input_id=args<0, 0>
  UDTF: foo(ColumnList<TextEncodingDict>, Column<int>, Column<TextEncodingDict>) -> Column<TextEncodingDict> | input_id=args<2> $=>$
        foo(ColumnListTextEncodingDict, ColumnInt32, ColumnTextEncodingDict) -> ColumnTextEncodingDict | input_id=args<2>
  UDTF: foo(Column<int>, Column<TextEncodingDict>) -> Column<TextEncodingDict> | input_id=args<1> $=>$
        foo(ColumnInt32, ColumnTextEncodingDict) -> ColumnTextEncodingDict | input_id=args<1>
  UDTF: foo(Column<int>, ColumnList<TextEncodingDict>) -> Column<TextEncodingDict> $=>$
        foo(ColumnInt32, ColumnListTextEncodingDict) -> ColumnTextEncodingDict | input_id=args<1, 0>
  UDTF: foo(Column<TextEncodingDict> | name = a) -> Column<TextEncodingDict> | name=out | input_id=args<0> $=>$
        foo(ColumnTextEncodingDict | name=a) -> ColumnTextEncodingDict | name=out | input_id=args<0>
  UDTF: foo(Column<TextEncodingDict> a) -> Column<TextEncodingDict> out | input_id=args<0> $=>$
        foo(ColumnTextEncodingDict | name=a) -> ColumnTextEncodingDict | name=out | input_id=args<0>

  UDTF: foo__cpu_template(Column<int32_t>) -> Column<int32_t> $=>$
        foo__cpu_template(ColumnInt32) -> ColumnInt32

  UDTF: foo__cpu(Column<T>, T, Cursor<ColumnList<U>>) -> Column<T>, T=[int32], U=[float] $=>$
        foo__cpu(ColumnInt32, Int32, Cursor<ColumnListFloat> | fields=[field0]) -> ColumnInt32
  UDTF: foo__cpu(Column<T> in1, T in2, Cursor<ColumnList<U>> in3) -> Column<T> out1, Column<U> out2, T=[int32], U=[float] $=>$
        foo__cpu(ColumnInt32 | name=in1, Int32 | name=in2, Cursor<ColumnListFloat> | name=in3 | fields=[field0]) -> ColumnInt32 | name=out1, ColumnFloat | name=out2

  UDTF: foo__cpu_template(Column<T>) -> Column<U>, T=[int32, int64], U=[float, double] $=>$
        foo__cpu_template(ColumnInt32) -> ColumnFloat  $=>$
        foo__cpu_template(ColumnInt64) -> ColumnFloat  $=>$
        foo__cpu_template(ColumnInt32) -> ColumnDouble $=>$
        foo__cpu_template(ColumnInt64) -> ColumnDouble

  UDTF: foo__cpu(TableFunctionManager, int64_t) -> Column<int64_t> $=>$
        foo__cpu(TableFunctionManager, Int64) -> ColumnInt64

  UDTF: foo__cpu(TableFunctionManager, Cursor<Column<int64_t> x, Column<int64_t> y> z) -> Column<int64_t> $=>$
        foo__cpu(TableFunctionManager, Cursor<ColumnInt64 | name=x, ColumnInt64 | name=y> | name=z | fields=[x,y]) -> ColumnInt64

  UDTF: foo__cpu(TableFunctionManager) | filter_table_function_transpose=on -> Column<int64_t> $=>$
        foo__cpu(TableFunctionManager) | filter_table_function_transpose=1 -> ColumnInt64
  UDTF: foo__cpu(TableFunctionManager) | filter_table_function_transpose=off -> Column<int64_t> $=>$
        foo__cpu(TableFunctionManager) | filter_table_function_transpose=0 -> ColumnInt64
  UDTF: foo__cpu(TableFunctionManager) | bar=off -> Column<int64_t> $=>$
        TransformerException: unknown function annotation: `bar`
  UDTF: foo__cpu(TableFunctionManager | bar=off) -> Column<int64_t> $=>$
        TransformerException: unknown input annotation: `bar`
  UDTF: foo__cpu(TableFunctionManager) -> Column<int64_t> | bar=off $=>$
        TransformerException: unknown output annotation: `bar`

  UDTF: foo__cpu(TableFunctionManager, Cursor<int32_t x> | fields=[x1]) -> Column<int64_t>!
        foo__cpu(TableFunctionManager, Cursor<Int32 | name=x> | fields=[x1]) -> ColumnInt64

  UDTF: foo_int_require(Column<int> | name=col, int32_t | require="sqrt(arg1) > col.size()" | name=arg1) -> int32_t $=>$
        foo_int_require(ColumnInt32 | name=col, Int32 | require="sqrt(arg1) > col.size()" | name=arg1) -> Int32
  UDTF: foo_int_require_mgr(TableFunctionManager, Column<int> | name=col, int32_t | require="sqrt(arg1) > col.size()" | name=arg1) -> int32_t $=>$
        foo_int_require_mgr(TableFunctionManager, ColumnInt32 | name=col, Int32 | require="sqrt(arg1) > col.size()" | name=arg1) -> Int32
  UDTF: foo_int_require(Column<int> col, int x | require="x > 0" | require="x < 5") -> int $=>$
        foo_int_require(ColumnInt32 | name=col, Int32 | name=x | require="x > 0" | require="x < 5") -> Int32
  UDTF: foo_str_require(TextEncodingNone s | require="s == \"str\"") -> int32_t $=>$
        foo_str_require(TextEncodingNone | name=s | require="s == \"str\"") -> Int32
  UDTF: foo_str_require(TextEncodingNone s | require="s != \"str\"") -> int32_t $=>$
        foo_str_require(TextEncodingNone | name=s | require="s != \"str\"") -> Int32


  UDTF: foo(TableFunctionManager, TextEncodingNone agg_type | require="agg_type == \"MAX\"",
            Cursor<Column<K> x, Column<T> y>) -> Column<int>, K=[int], T=[float] $=>$
        foo(TableFunctionManager, TextEncodingNone | name=agg_type | require="agg_type == \"MAX\"", Cursor<ColumnInt32 | name=x, ColumnFloat | name=y> | fields=[x,y]) -> ColumnInt32
 */
// clang-format on
