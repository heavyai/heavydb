import pytest
import pandas as pd
import numpy as np
import pandas.util.testing as tm

pytest.importorskip('rbc')


def catch_udf_support_disabled(mth):
    def new_mth(self, con):
        try:
            return mth(self, con)
        except Exception as msg:
            if type(
                msg
            ).__name__ == 'TOmniSciException' and msg.error_msg.startswith(
                'Runtime UDF registration is disabled'
            ):
                print('Ignoring `%s` failure' % (msg.error_msg))
                return
            raise

    new_mth.__name__ = mth.__name__
    return new_mth


@pytest.mark.usefixtures("omnisci_server")
class TestRuntimeUDF:
    def load_test_udf_incr(self, con):
        con.execute('drop table if exists test_udf_incr')
        con.execute('create table test_udf_incr (i4 integer, f8 double)')
        con.execute('insert into test_udf_incr values (1, 2.3);')
        con.execute('insert into test_udf_incr values (2, 3.4);')

    @catch_udf_support_disabled
    def test_udf_incr(self, con):
        @con('int32(int32)', 'double(double)')
        def incr(x):
            return x + 1

        self.load_test_udf_incr(con)

        result = list(con.execute('select i4, incr(i4) from test_udf_incr'))
        expected = [(1, 2), (2, 3)]
        assert result == expected

        result = list(con.execute('select f8, incr(f8) from test_udf_incr'))
        expected = [(2.3, 3.3), (3.4, 4.4)]
        assert result == expected

        con.execute('drop table if exists test_udf_incr')

    @catch_udf_support_disabled
    def test_udf_incr_read_sql(self, con):
        @con('int32(int32)', 'double(double)')
        def incr_read_sql(x):
            return x + 1

        self.load_test_udf_incr(con)

        result = pd.read_sql(
            '''select i4 as qty, incr_read_sql(i4) as price
               from test_udf_incr''',
            con,
        )
        expected = pd.DataFrame(
            {
                "qty": np.array([1, 2], dtype=np.int64),
                "price": np.array([2, 3], dtype=np.int64),
            }
        )[['qty', 'price']]
        tm.assert_frame_equal(result, expected)

        result = pd.read_sql(
            '''select f8 as qty, incr_read_sql(f8) as price
               from test_udf_incr''',
            con,
        )
        expected = pd.DataFrame(
            {
                "qty": np.array([2.3, 3.4], dtype=np.float64),
                "price": np.array([3.3, 4.4], dtype=np.float64),
            }
        )[['qty', 'price']]
        tm.assert_frame_equal(result, expected)

        con.execute('drop table if exists test_udf_incr')
