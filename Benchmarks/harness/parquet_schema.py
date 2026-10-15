# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read column types from a Parquet file footer (stdlib only, no pyarrow).

Parses just enough of the Thrift-compact FileMetaData to map each top-level leaf
column to the HeavyDB type that imports it without conversion.
"""

import struct

# Thrift compact protocol type ids
_STOP, _TRUE, _FALSE, _BYTE, _I16, _I32, _I64, _DOUBLE, _BINARY, _LIST, _SET, _MAP, _STRUCT = range(13)

# Parquet physical types
_BOOLEAN, _INT32, _INT64, _INT96, _FLOAT, _DOUBLE_T, _BYTE_ARRAY, _FIXED = range(8)
# ConvertedType values we care about
_CT_DECIMAL, _CT_DATE, _CT_INT_8, _CT_INT_16 = 5, 6, 15, 16


class _Reader:
    def __init__(self, buf):
        self.b, self.i = buf, 0

    def byte(self):
        v = self.b[self.i]
        self.i += 1
        return v

    def varint(self):
        r = shift = 0
        while True:
            c = self.byte()
            r |= (c & 0x7F) << shift
            if not c & 0x80:
                return r
            shift += 7

    def zigzag(self):
        n = self.varint()
        return (n >> 1) ^ -(n & 1)

    def value(self, t):
        if t == _TRUE:
            return True
        if t == _FALSE:
            return False
        if t == _BYTE:
            return self.byte()
        if t in (_I16, _I32, _I64):
            return self.zigzag()
        if t == _DOUBLE:
            v = struct.unpack("<d", self.b[self.i:self.i + 8])[0]
            self.i += 8
            return v
        if t == _BINARY:
            n = self.varint()
            v = bytes(self.b[self.i:self.i + n])
            self.i += n
            return v
        if t in (_LIST, _SET):
            h = self.byte()
            n, et = h >> 4, h & 0x0F
            if n == 15:
                n = self.varint()
            if et in (_TRUE, _FALSE):  # bool elements are one byte each
                return [self.byte() == 1 for _ in range(n)]
            return [self.value(et) for _ in range(n)]
        if t == _MAP:
            n = self.varint()
            if n == 0:
                return {}
            kv = self.byte()
            return {self.value(kv >> 4): self.value(kv & 0x0F) for _ in range(n)}
        if t == _STRUCT:
            return self.struct()
        raise ValueError("unsupported thrift type %d" % t)

    def struct(self):
        out, fid = {}, 0
        while True:
            h = self.byte()
            if h == _STOP:
                return out
            t = h & 0x0F
            delta = h >> 4
            fid = fid + delta if delta else self.zigzag()
            out[fid] = self.value(t)


def _heavy_type(el):
    """Map a SchemaElement dict to a HeavyDB type, or None if it should keep its declared type."""
    phys, conv = el.get(1), el.get(6)
    logical = el.get(10) or {}
    dec = logical.get(5)  # LogicalType.DECIMAL {1: scale, 2: precision}
    if conv == _CT_DECIMAL or dec:
        scale = el.get(7, dec.get(1) if dec else None)
        prec = el.get(8, dec.get(2) if dec else None)
        return "DECIMAL(%d,%d)" % (prec, scale) if prec is not None and scale is not None else None
    if phys == _INT32:
        return {_CT_DATE: "DATE", _CT_INT_8: "TINYINT", _CT_INT_16: "SMALLINT"}.get(conv, "INTEGER")
    if phys == _INT64:
        return "BIGINT" if conv is None else None  # timestamps etc.: keep declared type
    if phys == _FLOAT:
        return "FLOAT"
    if phys == _DOUBLE_T:
        return "DOUBLE"
    return None  # strings, booleans, nested: keep the benchmark schema's type


def read_column_types(path):
    """Return {column_name: heavydb_type_or_None} for a Parquet file, or None if unreadable."""
    try:
        with open(path, "rb") as f:
            f.seek(-8, 2)
            tail = f.read(8)
            if tail[4:] != b"PAR1":
                return None
            n = struct.unpack("<I", tail[:4])[0]
            f.seek(-8 - n, 2)
            meta = _Reader(f.read(n)).struct()
        elems = meta[2]
        cols, i = {}, 1  # elems[0] is the root
        while i < len(elems):
            el = elems[i]
            i += 1
            if el.get(5):  # nested group: skip its subtree, keep declared type
                pending = el[5]
                while pending and i < len(elems):
                    pending += elems[i].get(5, 0) - 1
                    i += 1
                cols[el[4].decode()] = None
            else:
                cols[el[4].decode()] = _heavy_type(el)
        return cols
    except (OSError, ValueError, KeyError, IndexError, struct.error):
        return None
