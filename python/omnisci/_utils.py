import datetime
import numpy as np


def seconds_to_time(seconds):
    """Convert seconds since midnight to a datetime.time"""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return datetime.time(h, m, s)


def time_to_seconds(time):
    """Convert a datetime.time to seconds since midnight"""
    if time is None:
        return None
    return 3600 * time.hour + 60 * time.minute + time.second


def datetime_in_precisions(epoch, precision):
    """Convert epoch time value into s, ms, us, ns"""
    base = datetime.datetime(1970, 1, 1)
    if precision == 0:
        return base + datetime.timedelta(seconds=epoch)
    elif precision == 3:
        seconds, modulus = divmod(epoch, 1000)
        return base + datetime.timedelta(seconds=seconds, milliseconds=modulus)
    elif precision == 6:
        seconds, modulus = divmod(epoch, 1000000)
        return base + datetime.timedelta(seconds=seconds, microseconds=modulus)
    elif precision == 9:
        return np.datetime64(epoch, 'ns')
    else:
        raise TypeError("Invalid timestamp precision: {}".format(precision))


def date_to_seconds(arr):
    """Converts date into seconds"""

    return arr.apply(lambda x: np.datetime64(x, "s").astype(int))


mapd_to_slot = {
    'BOOL': 'int_col',
    'BOOLEAN': 'int_col',
    'SMALLINT': 'int_col',
    'INT': 'int_col',
    'INTEGER': 'int_col',
    'BIGINT': 'int_col',
    'FLOAT': 'real_col',
    'DECIMAL': 'int_col',
    'DOUBLE': 'real_col',
    'TIMESTAMP': 'int_col',
    'DATE': 'int_col',
    'TIME': 'int_col',
    'STR': 'str_col',
    'TINYINT': 'int_col',
}


mapd_to_na = {
    'BOOL': -128,
    'BOOLEAN': -128,
    'SMALLINT': -32768,
    'INT': -2147483648,
    'INTEGER': -2147483648,
    'BIGINT': -9223372036854775808,
    'FLOAT': 0,
    'DECIMAL': 0,
    'DOUBLE': 0,
    'TIMESTAMP': -9223372036854775808,
    'DATE': -9223372036854775808,
    'TIME': -9223372036854775808,
    'STR': '',
    'TINYINT': -128,
}
