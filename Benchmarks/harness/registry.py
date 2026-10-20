# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark discovery and the BenchmarkDef description loaded from benchmark.toml."""

import re
import string
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover
    sys.exit("Python >= 3.11 is required (tomllib)")

BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks"


@dataclass
class Table:
    name: str
    path: str  # relative to the dataset root; a directory of parquet files or a file
    glob: str = "*.parquet"


@dataclass
class BenchmarkDef:
    name: str
    description: str
    directory: Path
    tables: list
    schema_file: str = "schema.sql"
    query_dir: str = "queries"
    query_glob: str = "*.sql"
    iterations: int = 3
    warmup: int = 1
    disabled: dict = field(default_factory=dict)   # query name -> reason
    setup_sql: list = field(default_factory=list)
    teardown_sql: list = field(default_factory=list)
    vars: dict = field(default_factory=dict)       # ${NAME} substitutions in the schema file and queries

    def schema_statements(self):
        text = (self.directory / self.schema_file).read_text()  # absolute paths win
        text = string.Template(text).substitute(self.vars)
        text = re.sub(r"--.*", "", text)
        return {m.group(1): m.group(0).strip()
                for m in re.finditer(r"CREATE\s+TABLE\s+(\w+)\s*\(.*?\)\s*(?:WITH\s*\(.*?\))?\s*;",
                                     text, re.S | re.I)}

    def render_query(self, name, text, extra_vars=None):
        """Substitute ${NAME} in query text from [vars] plus run-time values such as ${SF}."""
        try:
            return string.Template(text).substitute({**self.vars, **(extra_vars or {})})
        except KeyError as e:
            raise SystemExit("query %s uses ${%s}, which is not set (dataset metadata.json "
                             "scale_factor, a [vars] entry, or --var %s=VALUE)" % (name, e.args[0], e.args[0]))

    def queries(self):
        files = sorted((self.directory / self.query_dir).glob(self.query_glob))
        return [(f.stem, f.read_text()) for f in files]


def load(name):
    d = BENCH_DIR / name
    f = d / "benchmark.toml"
    if not f.is_file():
        raise SystemExit("unknown benchmark %r; available: %s" % (name, ", ".join(names())))
    cfg = tomllib.loads(f.read_text())
    tables = [Table(t["name"], t.get("path", t["name"]), t.get("glob", "*.parquet"))
              for t in cfg["tables"]]
    return BenchmarkDef(
        name=cfg.get("name", name), description=cfg.get("description", ""), directory=d,
        tables=tables, schema_file=cfg.get("schema_file", "schema.sql"),
        query_dir=cfg.get("query_dir", "queries"), query_glob=cfg.get("query_glob", "*.sql"),
        iterations=cfg.get("iterations", 3), warmup=cfg.get("warmup", 1),
        disabled=cfg.get("disabled", {}), setup_sql=cfg.get("setup_sql", []),
        teardown_sql=cfg.get("teardown_sql", []),
        vars={k: str(v) for k, v in cfg.get("vars", {}).items()})


def names():
    return sorted(p.parent.name for p in BENCH_DIR.glob("*/benchmark.toml"))
