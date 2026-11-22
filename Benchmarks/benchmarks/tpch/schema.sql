-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0
-- TPC-H schema (decimal types; precision comes from [vars] DEC_PRECISION, which must
-- match the dataset: 15 for most generators, 12 for gqe-dbgen).  One CREATE TABLE per table; the importer
-- derives the foreign-table DDL from these statements.
-- Columns used in GROUP BY / correlated subqueries are dictionary-encoded (HeavyDB
-- cannot group by none-encoded text); free-text columns stay ENCODING NONE.

CREATE TABLE region (
  r_regionkey BIGINT,
  r_name TEXT ENCODING DICT(32),
  r_comment TEXT ENCODING NONE
);

CREATE TABLE nation (
  n_nationkey BIGINT,
  n_name TEXT ENCODING DICT(32),
  n_regionkey BIGINT,
  n_comment TEXT ENCODING NONE
);

CREATE TABLE supplier (
  s_suppkey BIGINT,
  s_name TEXT ENCODING DICT(32),
  s_address TEXT ENCODING NONE,
  s_nationkey BIGINT,
  s_phone TEXT ENCODING NONE,
  s_acctbal DECIMAL(${DEC_PRECISION},2),
  s_comment TEXT ENCODING NONE
);

CREATE TABLE customer (
  c_custkey BIGINT,
  c_name TEXT ENCODING DICT(32),
  c_address TEXT ENCODING DICT(32),
  c_nationkey BIGINT,
  c_phone TEXT ENCODING DICT(32),
  c_acctbal DECIMAL(${DEC_PRECISION},2),
  c_mktsegment TEXT ENCODING DICT(32),
  c_comment TEXT ENCODING DICT(32)
);

CREATE TABLE part (
  p_partkey BIGINT,
  p_name TEXT ENCODING NONE,
  p_mfgr TEXT ENCODING DICT(32),
  p_brand TEXT ENCODING DICT(32),
  p_type TEXT ENCODING DICT(32),
  p_size INTEGER,
  p_container TEXT ENCODING DICT(32),
  p_retailprice DECIMAL(${DEC_PRECISION},2),
  p_comment TEXT ENCODING NONE
);

CREATE TABLE partsupp (
  ps_partkey BIGINT,
  ps_suppkey BIGINT,
  ps_availqty INTEGER,
  ps_supplycost DECIMAL(${DEC_PRECISION},2),
  ps_comment TEXT ENCODING NONE
);

CREATE TABLE orders (
  o_orderkey BIGINT,
  o_custkey BIGINT,
  o_orderstatus TEXT ENCODING DICT(32),
  o_totalprice DECIMAL(${DEC_PRECISION},2),
  o_orderdate DATE,
  o_orderpriority TEXT ENCODING DICT(32),
  o_clerk TEXT ENCODING NONE,
  o_shippriority INTEGER,
  o_comment TEXT ENCODING NONE
);

CREATE TABLE lineitem (
  l_orderkey BIGINT,
  l_partkey BIGINT,
  l_suppkey BIGINT,
  l_linenumber INTEGER,
  l_quantity DECIMAL(${DEC_PRECISION},2),
  l_extendedprice DECIMAL(${DEC_PRECISION},2),
  l_discount DECIMAL(${DEC_PRECISION},2),
  l_tax DECIMAL(${DEC_PRECISION},2),
  l_returnflag TEXT ENCODING DICT(32),
  l_linestatus TEXT ENCODING DICT(32),
  l_shipdate DATE,
  l_commitdate DATE,
  l_receiptdate DATE,
  l_shipinstruct TEXT ENCODING DICT(32),
  l_shipmode TEXT ENCODING DICT(32),
  l_comment TEXT ENCODING NONE
);
