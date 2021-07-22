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

  UDTF: foo(Column<TextEncodedDict>) -> Column<TextEncodedDict> !
        foo(ColumnTextEncodedDict) -> ColumnTextEncodedDict | input_id=args<0>
  UDTF: foo(Column<TextEncodedDict>) | name = a -> Column<TextEncodedDict> | name=out | input_id=args<0> !
        foo(ColumnTextEncodedDict) -> ColumnTextEncodedDict | name=out | input_id=args<0>

  UDTF: foo__cpu_template(Column<int32_t>) -> Column<int32_t> !
        foo__cpu_template(ColumnInt32) -> ColumnInt32
 */
// clang-format on
