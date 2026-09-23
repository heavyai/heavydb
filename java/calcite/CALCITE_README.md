# HeavyDB Calcite Integration

## Overview

HeavyDB uses Apache Calcite as its SQL parser, validator, and query planner. The
integration lives under `java/calcite/src/main/java/` and is built as a standalone
JAR that the C++ server loads as a separate process/server.

The source tree contains two distinct types of Java file:

| Package root | Type | Purpose                                                                                                                                |
|---|---|----------------------------------------------------------------------------------------------------------------------------------------|
| `com/mapd/` | HeavyDB first-party code | Parser driver, operator tables, metadata layer, rel-JSON serialisation                                                                 |
| `org/apache/calcite/` | Calcite-namespace files | Overlays that shadow upstream Calcite classes, plus HeavyDB additions placed in the Calcite package for subclassing and package access |

There are ~120 files under `com/mapd/` and ~30 under `org/apache/calcite/`.

---

## Calcite overlay mechanism

HeavyDB patches Calcite by placing modified copies of upstream source files at the
same package path under `org/apache/calcite/`.

Modifications inside overlaid files should be wrapped with block comments:

```java
// HEAVY.AI START BLOCK: <reason>
...modified or added code...
// HEAVY.AI END BLOCK
```

Single-line substitutions use an inline comment:

```java
SomeUpstreamValue, // HEAVY.AI: upstream uses OtherValue
```

When upgrading Calcite, these markers are the primary guide to which sections need
re-applying to the new upstream source.

---

## True Calcite overlays

These 13 files shadow an upstream source file of the same package and name. The
baseline is Calcite 1.41.0 (`calcite-core`, pinned in `pom.xml`), except for
`avatica/util/DateTimeUtils.java`, whose upstream is Avatica 1.27.0
(`avatica-core`) — Avatica is a separate Apache project with its own release
cycle, so that file is not found in a Calcite source tree.

The **Markers** column is the count of case-insensitive `heavy`/`mapd` hits.
Treat it as a lower bound only, not a measure of how much a file diverges:
`sql/fun/SqlArrayValueConstructor.java` and `sql/fun/SqlStdOperatorTable.java`
both carry substantial modifications with **zero** markers, and
`sql/type/SqlTypeFactoryImpl.java` and `sql/fun/SqlSingleValueAggFunction.java`
are modified purely by deletion, which leaves nothing to mark. Diff against the
pinned upstream release when upgrading rather than relying on these counts.

| File (relative to `org/apache/calcite/`) | Markers | Key HeavyDB change |
|---|---|---|
| `sql2rel/SqlToRelConverter.java` | 32 | Subquery expansion decided per-site by `HeavyDBSqlToRelConverter.Config.getExpandPredicate()` instead of the blanket `config.isExpand()`, with `Blackboard.getTopNode()` added to supply the predicate's left argument; `getColumnMappings` visibility widened to `protected` so `HeavyDBSqlToRelConverter` can override it and handle `ExtTableFunction` column mappings for filter pushdown; `convertCall` rewrites `ExtTableFunction` calls to fill in omitted default arguments; `createAggregate` rebuilt on `LogicalAggregate.create` with `ImmutableBitSet.ORDERING`-sorted group sets in place of `RelBuilder.aggregate` |
| `sql/validate/SqlValidatorImpl.java` | 10 | `TypeCoercions.createHeavyDBTypeCoercion` used in place of `config.typeCoercionFactory().create`; DECIMAL literals exceeding max precision or scale fall back to `validateLiteralAsDouble` instead of raising `numberLiteralOutOfRange`; `requiresSpecialCreate` injection in `visitScoped` for `HeavyDBDefaultArgInjector` default arguments; GROUP BY ordinal expansion fix to fire the same injection before type-checking |
| `avatica/util/DateTimeUtils.java` | 5 | Sole change: `validateLenientDate` accepts year-only date strings, so `2000` parses as `2000-01-01` (the parsing path in `dateStringToUnixDate` already handled this; only validation needed to pass it through). Upstream is Avatica, not Calcite — see the note above |
| `rex/RexSimplify.java` | 5 | Division by a literal with a positive scale falls through to `simplifyGenericNode` rather than the `oneIndex` shortcut; casts of `BigDecimal` literals to DECIMAL or integer targets are left unfolded so HeavyDB's own overflow checks still apply |
| `sql/validate/implicit/TypeCoercions.java` | 8 | `createHeavyDBTypeCoercion` factory added, returning `HeavyDBTypeCoercion`; the upstream factory methods are untouched. This is the hook `SqlValidatorImpl` calls |
| `prepare/CalciteSqlValidator.java` | 6 | `addToSelectList` overridden, with an `isSystemColumn` helper resolving the identifier against `HeavyDBTable`, so HeavyDB system columns stay out of `SELECT *` expansion; class and constructor narrowed from `public` to package-private |
| `prepare/PlannerImpl.java` | 6 | `createSqlToRelConverter` factory method added (`protected`) so `HeavyDBPlanner` can substitute `HeavyDBSqlToRelConverter` via virtual dispatch; `createRexBuilder` widened from `private` to `protected` so it can likewise substitute `HeavyDBRexBuilder` |
| `rel/rules/DynamicFilterJoinRule.java` | 4 | HeavyDB dynamic filter join rule modifications |
| `sql/type/SqlTypeFactoryImpl.java` | 2 | Deletion only: the precision-widening block is removed from `leastRestrictiveSqlType`, so two operands sharing a type name but differing in precision no longer widen the result to the larger precision |
| `sql2rel/StandardConvertletTable.java` | 6 | `TryCast` branch added to `convertCall` that strips `SqlKind.OTHER` operands from `HeavyDBSqlOperatorTable.TryCast` calls |
| `sql/fun/SqlSingleValueAggFunction.java` | 2 | Deletion only: the `getDistinctOptionality()` override is removed, so SINGLE_VALUE no longer reports `Optionality.IGNORED` for DISTINCT |
| `sql/fun/SqlArrayValueConstructor.java` | 0 | DECIMAL-to-DOUBLE promotion in array constructors when literal scale exceeds the component type's scale at max precision, via the added `shouldPromoteToApproximate` helper. **Unmarked — no `HEAVY.AI` comments** |
| `sql/fun/SqlStdOperatorTable.java` | 0 | Set-op operand type checker that rejects mismatched typed-null integer casts, restoring Calcite 1.25 strictness; applied to UNION, EXCEPT and INTERSECT plus their ALL forms via the added `hasMismatchedSetOpTypedNullIntegerCast`. `NUMERIC_INTEGER` used in place of upstream `NUMERIC_INT32` for ROUND and TRUNCATE, accepting any integer width. **Unmarked — no `HEAVY.AI` comments** |

---

## HeavyDB additions in the Calcite namespace

These 17 files have no upstream Calcite counterpart. They are placed under
`org/apache/calcite/` solely to gain package-private access to Calcite internals.

| File (relative to `org/apache/calcite/`) | Purpose |
|---|---|
| `prepare/HeavyDBPlanner.java` | Subclass of `PlannerImpl`; overrides `createSqlToRelConverter` to inject `HeavyDBSqlToRelConverter`, and `createRexBuilder` to inject `HeavyDBRexBuilder` |
| `prepare/HeavyDBSqlAdvisor.java` | SQL completion advisor |
| `prepare/HeavyDBSqlAdvisorValidator.java` | Validator used by the SQL advisor |
| `prepare/SqlIdentifierCapturer.java` | Captures SQL identifiers during parsing for schema resolution |
| `rel/externalize/HeavyDBRelJson.java` | HeavyDB extensions to Calcite's rel-node JSON serialisation |
| `rel/externalize/HeavyDBRelJsonReader.java` | Reads HeavyDB rel-node JSON into a `RelNode` tree |
| `rel/externalize/HeavyDBRelJsonWriter.java` | Writes a `RelNode` tree to HeavyDB rel-node JSON |
| `rel/externalize/HeavyDBRelWriterImpl.java` | Implementation detail of the rel-JSON writer |
| `rel/rules/InjectFilterRule.java` | Injects row-level security filters into the plan |
| `rel/rules/OuterJoinOptViaNullRejectionRule.java` | Converts outer joins to inner joins where nulls are rejected by a subsequent predicate |
| `rel/rules/QueryOptimizationRules.java` | HeavyDB-specific planner rule utilities |
| `rel/rules/Restriction.java` | Data class carrying row-level security restriction predicates |
| `rex/HeavyDBRexBuilder.java` | Subclass of `RexBuilder`; overrides `makeCast` and `canRemoveCastFromLiteral` |
| `rex/HeavyDBRexExecutor.java` | HeavyDB Rex expression executor |
| `sql/HeavyDBDefaultArgInjector.java` | Marker interface for `SqlOperator` subclasses that inject default arguments via `createCall()` when optional operands are omitted; `requiresSpecialCreate()` signals when injection is needed |
| `sql2rel/HeavyDBSqlToRelConverter.java` | Subclass of `SqlToRelConverter`; overrides `getColumnMappings` to handle `ExtTableFunction` (UDTFs) for filter pushdown into CURSOR arguments |
| `util/EscapedStringJsonBuilder.java` | JSON builder with HeavyDB-specific string escaping |

---
