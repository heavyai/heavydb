.. OmniSciDB Data Model

==================================
Data Types
==================================

OmniSciDB supports a variety of data types, including scalar types with an optional encoding and variable length types. The full list of data types is available on in OmniSciDB `user facing documentation <https://docs.omnisci.com/latest/5_datatypes.html#fixed-encoding>`_.

Scalar Types
------------

Scalar types (e.g. `INT`, `DOUBLE`, `DATE`) are stored in compact buffers using the smallest possible size. E.g., the data buffer for an integer column would store each entry in a 4 byte "slot", and the `n`th entry could be found by incrementing the pointer to the start of the buffer by `n * 4` bytes. 

Scalar Types with Encoding
--------------------------

OmniSciDB supports an optional `encoding` parameter for most scalar types. An `encoding` allows a type to be stored with fewer bytes than would otherwise be required, typically by limiting the range of the type. For example, `DATE` columns can be encoded in `DAYS` (instead of `SECONDS`, the default for the scalar `DATE` type) using the syntax `DATE ENCODING DAYS(16)` (note that in OmniSciDB DDL, while most types default to "none encoding" if no encoding is specified, `DATE` defaults to encoded in days using 32-bits). The encoded data is left in encoded form until it is read from the in-memory buffer for purposes of manipulation during a query. Thus, `DATE ENCODING DAYS(16)` will be converted on the fly from the number of days since the unix epoch in a 16-bit integer to the number of seconds since unix epoch in a 64-bit integer. The decoded value typically lives in a register and a decoded buffer is typically never created in main memory. 

Note that we use the term `encoding` and not `compression` since the encoding is applied **per-value**, and not the entire buffer. An encoded buffer still supports random access without transformation of the entire buffer. 

Variable Length Types
---------------------

Variable length data types (arrays and none-encoded strings) consist of two buffers; an index buffer and a data buffer. The index buffer specifies an offset into the data buffer for the given row; that is, each row in the index buffer has a fixed size, whereas each row in the data buffer has a varying size. During query execution, the value from the index buffer for the given row is read first (since the index buffer supports random access without a scan), and then the varlen payload is loaded from the data buffer.