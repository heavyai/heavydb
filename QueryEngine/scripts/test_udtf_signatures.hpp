/*
  This file contains test inputs to generate_TableFunctionsFactory_init.py script.
*/

// clang-format off
/*

  UDTF: foo(Column<int32>) -> Column<int32> !
        foo(ColumnInt32) -> ColumnInt32

  UDTF: foo(Column<int32>, RowMultiplier) -> Column<int32> !
        foo(ColumnInt32, kUserSpecifiedRowMultiplier<2>) -> ColumnInt32

  UDTF: foo(RowMultiplier, Column<int32>) -> Column<int32> !
        foo(kUserSpecifiedRowMultiplier<1>, ColumnInt32) -> ColumnInt32

  UDTF: foo(Column<int32>, Constant<5>) -> Column<int32> !
        foo(ColumnInt32, kConstant<5>) -> ColumnInt32

  UDTF: foo(Cursor<Column<int32>>) -> Column<int32> !
        foo(Cursor<ColumnInt32>) -> ColumnInt32

  UDTF: foo(Cursor<Column<int32>, Column<float>>) -> Column<int32> !
        foo(Cursor<ColumnInt32, ColumnFloat>) -> ColumnInt32
  UDTF: foo(Cursor<Column<int32>>, Cursor<Column<float>>) -> Column<int32> !
        foo(Cursor<ColumnInt32>, Cursor<ColumnFloat>) -> ColumnInt32

  UDTF: foo(Column<int32>) -> Column<int32>, Column<float> !
        foo(ColumnInt32) -> ColumnInt32, ColumnFloat
  UDTF: foo(Column<int32>|name=a) -> Column<int32>|name=out !
        foo(ColumnInt32 | name=a) -> ColumnInt32 | name=out

  UDTF: foo(Column<TextEncodingDict>) -> Column<TextEncodingDict> !
        foo(ColumnTextEncodingDict) -> ColumnTextEncodingDict | input_id=args<0>
  UDTF: foo(ColumnList<TextEncodingDict>) -> Column<TextEncodingDict> !
        foo(ColumnListTextEncodingDict) -> ColumnTextEncodingDict | input_id=args<0, 0>
  UDTF: foo(ColumnList<TextEncodingDict>, Column<int>, Column<TextEncodingDict>) -> Column<TextEncodingDict> | input_id=args<2> !
        foo(ColumnListTextEncodingDict, ColumnInt32, ColumnTextEncodingDict) -> ColumnTextEncodingDict | input_id=args<2>
  UDTF: foo(Column<int>, Column<TextEncodingDict>) -> Column<TextEncodingDict> | input_id=args<1> !
        foo(ColumnInt32, ColumnTextEncodingDict) -> ColumnTextEncodingDict | input_id=args<1>
  UDTF: foo(Column<int>, ColumnList<TextEncodingDict>) -> Column<TextEncodingDict> !
        foo(ColumnInt32, ColumnListTextEncodingDict) -> ColumnTextEncodingDict | input_id=args<1, 0>
  UDTF: foo(Column<TextEncodingDict> | name = a) -> Column<TextEncodingDict> | name=out | input_id=args<0> !
        foo(ColumnTextEncodingDict | name=a) -> ColumnTextEncodingDict | name=out | input_id=args<0>

  UDTF: foo__cpu_template(Column<int32_t>) -> Column<int32_t> !
        foo__cpu_template(ColumnInt32) -> ColumnInt32

  UDTF: foo__cpu(Column<T>, T, Cursor<ColumnList<U>>) -> Column<T>, T=[int32], U=[float] !
        foo__cpu(ColumnInt32, Int32, Cursor<ColumnListFloat>) -> ColumnInt32

  UDTF: foo__cpu_template(Column<T>) -> Column<U>, T=[int32, int64], U=[float, double] !
        foo__cpu_template(ColumnInt32) -> ColumnFloat  !
        foo__cpu_template(ColumnInt64) -> ColumnFloat  !
        foo__cpu_template(ColumnInt32) -> ColumnDouble !
        foo__cpu_template(ColumnInt64) -> ColumnDouble

 */
// clang-format on
