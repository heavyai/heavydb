from omnisci.cursor import make_row_results_set
from omnisci.thrift.ttypes import (
    TRowSet,
    TColumnType,
    TColumn,
    TColumnData,
    TQueryResult,
)
from omnisci.common.ttypes import TTypeInfo


class TestRowResults:
    def test_nulls_handled(self):

        rs = TQueryResult(
            TRowSet(
                row_desc=[
                    TColumnType(
                        col_name='a', col_type=TTypeInfo(type=0, nullable=True)
                    ),
                    TColumnType(
                        col_name='b', col_type=TTypeInfo(type=1, nullable=True)
                    ),
                    TColumnType(
                        col_name='c', col_type=TTypeInfo(type=2, nullable=True)
                    ),
                    TColumnType(
                        col_name='d', col_type=TTypeInfo(type=3, nullable=True)
                    ),
                    TColumnType(
                        col_name='e', col_type=TTypeInfo(type=4, nullable=True)
                    ),
                    TColumnType(
                        col_name='f', col_type=TTypeInfo(type=5, nullable=True)
                    ),
                    TColumnType(
                        col_name='g', col_type=TTypeInfo(type=6, nullable=True)
                    ),
                    TColumnType(
                        col_name='h', col_type=TTypeInfo(type=7, nullable=True)
                    ),
                    TColumnType(
                        col_name='i', col_type=TTypeInfo(type=8, nullable=True)
                    ),
                    TColumnType(
                        col_name='j', col_type=TTypeInfo(type=9, nullable=True)
                    ),
                    TColumnType(
                        col_name='k',
                        col_type=TTypeInfo(type=10, nullable=True),
                    ),
                ],
                rows=[],
                columns=[
                    TColumn(
                        data=TColumnData(int_col=[-2147483648]), nulls=[True]
                    ),
                    TColumn(
                        data=TColumnData(int_col=[-2147483648]), nulls=[True]
                    ),
                    TColumn(
                        data=TColumnData(int_col=[-2147483648]), nulls=[True]
                    ),
                    TColumn(
                        data=TColumnData(real_col=[-2147483648]), nulls=[True]
                    ),  # noqa
                    TColumn(
                        data=TColumnData(real_col=[-2147483648]), nulls=[True]
                    ),  # noqa
                    TColumn(
                        data=TColumnData(real_col=[-2147483648]), nulls=[True]
                    ),  # noqa
                    TColumn(
                        data=TColumnData(str_col=[-2147483648]), nulls=[True]
                    ),
                    TColumn(
                        data=TColumnData(int_col=[-2147483648]), nulls=[True]
                    ),
                    TColumn(
                        data=TColumnData(int_col=[-2147483648]), nulls=[True]
                    ),
                    TColumn(
                        data=TColumnData(int_col=[-2147483648]), nulls=[True]
                    ),
                    TColumn(
                        data=TColumnData(int_col=[-2147483648]), nulls=[True]
                    ),
                ],
                is_columnar=True,
            )
        )

        result = list(make_row_results_set(rs))
        assert result == [(None,) * 11]
