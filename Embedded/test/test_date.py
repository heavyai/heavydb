import sys
import pytest
import pyarrow as pa
import omniscidbe as dbe
import ctypes
ctypes._dlopen('libDBEngine.so', ctypes.RTLD_GLOBAL)

sys.setdlopenflags(1 | 256)

def test_init():
    global engine
    engine = dbe.PyDbEngine()
    assert bool(engine.closed) == False

engine = None


@pytest.mark.parametrize(
    "date",
    [
        (pa.date32, "date32"),
        (pa.date64, "date64")
    ],
)
def test_date(date):
    table = pa.Table.from_pydict({"a": [1, 2, 3, 4]}, schema=pa.schema({"a": date[0]()}))
    assert table
    test_name = "test_table_{}".format(date[1])
    engine.importArrowTable(test_name, table)
    assert bool(engine.closed) == False
    cursor = engine.executeDML("select * from {}".format(test_name))
    assert cursor
    batch = cursor.getArrowRecordBatch()
    assert batch


if __name__ == "__main__":
    pytest.main(["-v", __file__])
