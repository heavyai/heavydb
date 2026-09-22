/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package org.apache.calcite.prepare;

import com.google.common.collect.ImmutableSet;
import com.mapd.calcite.parser.HeavyDBParserOptions;
import com.mapd.calcite.parser.HeavyDBSchema;
import com.mapd.calcite.parser.ProjectProjectRemoveRule;
import com.mapd.calcite.rel.rules.HeavyDBAggregateOuterJoinStrengthReductionRule;
import com.mapd.calcite.rel.rules.FilterTableFunctionMultiInputTransposeRule;
import com.mapd.calcite.rel.rules.HeavyDBFilteredUniqueKeysetJoinRule;
import com.mapd.calcite.rel.rules.HeavyDBAggregateJoinPayloadDeferralRule;
import com.mapd.calcite.rel.rules.HeavyDBAggregateJoinPayloadKeysetRule;
import com.mapd.calcite.rel.rules.HeavyDBAggregateJoinReductionRule;
import com.mapd.calcite.rel.rules.HeavyDBAggregateJoinWindowExtremaRule;
import com.mapd.calcite.rel.rules.HeavyDBConnectedJoinOptimizeRule;
import com.mapd.calcite.rel.rules.HeavyDBDifferentValueAggregateJoinRule;
import com.mapd.calcite.rel.rules.HeavyDBDistinctAggregateJoinPruneRule;
import com.mapd.calcite.rel.rules.HeavyDBExistenceCountToGroupByRule;
import com.mapd.calcite.rel.rules.HeavyDBJoinTreeKeysetReductionRule;
import com.mapd.calcite.rel.rules.HeavyDBJoinFilterSplitRule;
import com.mapd.calcite.rel.rules.HeavyDBKeyPreservingAggregateRule;
import com.mapd.calcite.rel.rules.HeavyDBLargeFilteredTableJoinRule;
import com.mapd.calcite.rel.rules.HeavyDBLeftJoinAntiSemiJoinRule;
import com.mapd.calcite.rel.rules.HeavyDBLeftJoinCountDistributionRule;
import com.mapd.calcite.rel.rules.HeavyDBNotInToAntiJoinRule;
import com.mapd.calcite.rel.rules.HeavyDBOuterJoinMultiJoinDecomposeRule;
import com.mapd.calcite.rel.rules.HeavyDBPairedDifferentValueStatsRule;
import com.mapd.calcite.rel.rules.HeavyDBRedundantSemiJoinPruneRule;
import com.mapd.calcite.rel.rules.HeavyDBScalarExtremaJoinToTopNRule;
import com.mapd.calcite.rel.rules.HeavyDBSingleCountDistinctToGroupByRule;

import org.apache.calcite.config.CalciteConnectionConfig;
import org.apache.calcite.config.CalciteConnectionConfigImpl;
import org.apache.calcite.config.CalciteConnectionProperty;
import org.apache.calcite.jdbc.CalciteSchema;
import org.apache.calcite.linq4j.function.Functions;
import org.apache.calcite.plan.Context;
import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.prepare.Prepare;
import org.apache.calcite.plan.RelOptCostImpl;
import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.hep.HepPlanner;
import org.apache.calcite.plan.hep.HepProgram;
import org.apache.calcite.plan.hep.HepProgramBuilder;
import org.apache.calcite.plan.volcano.VolcanoPlanner;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rel.externalize.HeavyDBRelJsonReader;
import org.apache.calcite.rel.rules.*;
import org.apache.calcite.DataContexts;
import org.apache.calcite.rex.HeavyDBRexBuilder;
import org.apache.calcite.rex.HeavyDBRexExecutor;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.sql2rel.HeavyDBSqlToRelConverter;
import org.apache.calcite.sql2rel.SqlRexConvertletTable;
import org.apache.calcite.sql2rel.SqlToRelConverter;
import org.apache.calcite.schema.SchemaPlus;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.advise.SqlAdvisor;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.calcite.sql.validate.SqlConformanceEnum;
import org.apache.calcite.sql.validate.SqlMoniker;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.tools.FrameworkConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

/**
 * Customised version of the PlannerImpl for HEAVY.AI. Used to be a copy of
 * PlannerImpl, refactored now to use inheritance to minimize maintenance
 * efforts. Implementation of {@link org.apache.calcite.tools.Planner}.
 */
public class HeavyDBPlanner extends PlannerImpl {
  FrameworkConfig config;
  private List<HeavyDBParserOptions.FilterPushDownInfo> filterPushDownInfo =
          new ArrayList<>();
  private List<Restriction> restrictions = null;
  final static Logger HEAVYDBLOGGER = LoggerFactory.getLogger(HeavyDBPlanner.class);

  public HeavyDBPlanner(FrameworkConfig config) {
    super(config);
    this.config = config;
  }

  // Install HeavyDBRexBuilder for the SQL-text path. Without this override,
  // PlannerImpl.rel() and PlannerImpl.expandView() use the stock RexBuilder
  // and our HeavyDBRexBuilder.makeCast / canRemoveCastFromLiteral overrides
  // never fire (they would only fire on the RA-from-JSON path which already
  // explicitly constructs new HeavyDBRexBuilder(...) below).
  @Override
  protected RexBuilder createRexBuilder() {
    return new HeavyDBRexBuilder(getTypeFactory());
  }

  private static SchemaPlus rootSchema(SchemaPlus schema) {
    for (;;) {
      if (schema.getParentSchema() == null) {
        return schema;
      }
      schema = schema.getParentSchema();
    }
  }

  public static class CompletionResult {
    public List<SqlMoniker> hints;
    public String replaced;

    CompletionResult(final List<SqlMoniker> hints, final String replaced) {
      this.hints = hints;
      this.replaced = replaced;
    }
  }

  private CalciteCatalogReader createCatalogReader() {
    final SchemaPlus rootSchema = rootSchema(config.getDefaultSchema());
    final Context context = config.getContext();
    final CalciteConnectionConfig connectionConfig;

    if (context != null) {
      connectionConfig = context.unwrap(CalciteConnectionConfig.class);
    } else {
      Properties properties = new Properties();
      properties.setProperty(CalciteConnectionProperty.CASE_SENSITIVE.camelName(),
              String.valueOf(config.getParserConfig().caseSensitive()));
      connectionConfig = new CalciteConnectionConfigImpl(properties);
    }

    return new CalciteCatalogReader(CalciteSchema.from(rootSchema),
            CalciteSchema.from(config.getDefaultSchema()).path(null),
            getTypeFactory(),
            connectionConfig);
  }

  public void advanceToValidate() {
    try {
      String dummySql = "SELECT 1";
      super.parse(dummySql);
    } catch (SqlParseException e) {
      throw new RuntimeException(e);
    }
  }

  public void ready() {
    // need to call ready on the super class, but that method is marked private
    // circumventing via reflection for now
    try {
      Method readyMethod = getClass().getSuperclass().getDeclaredMethod("ready");
      readyMethod.setAccessible(true);
      readyMethod.invoke(this);
    } catch (InvocationTargetException e) {
      if (e.getCause() instanceof RuntimeException) {
        throw(RuntimeException) e.getCause();
      } else {
        throw new RuntimeException(e.getCause());
      }
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  public CompletionResult getCompletionHints(
          final String sql, final int cursor, final List<String> visibleTables) {
    ready();

    SqlValidator.Config validatorConfig = SqlValidator.Config.DEFAULT;
    validatorConfig = validatorConfig.withSqlConformance(SqlConformanceEnum.LENIENT);

    HeavyDBSqlAdvisorValidator advisor_validator =
            new HeavyDBSqlAdvisorValidator(visibleTables,
                    config.getOperatorTable(),
                    createCatalogReader(),
                    getTypeFactory(),
                    validatorConfig);
    SqlAdvisor advisor =
            new HeavyDBSqlAdvisor(advisor_validator, config.getParserConfig());
    String[] replaced = new String[1];
    int adjusted_cursor = cursor < 0 ? sql.length() : cursor;
    java.util.List<SqlMoniker> hints =
            advisor.getCompletionHints(sql, adjusted_cursor, replaced);
    return new CompletionResult(hints, replaced[0]);
  }

  public static HepPlanner getHepPlanner(
          HepProgram hepProgram, boolean doNotEliminateSharedNodesInQueryPlanDag) {
    final HepPlanner planner;
    if (doNotEliminateSharedNodesInQueryPlanDag) {
      planner = new HepPlanner(
              hepProgram, null, true, Functions.ignore2(), RelOptCostImpl.FACTORY);
    } else {
      planner = new HepPlanner(hepProgram);
    }
    planner.setExecutor(new HeavyDBRexExecutor(DataContexts.EMPTY));
    return planner;
  }

  // Override the PlannerImpl factory method so that virtual dispatch causes
  // HeavyDBSqlToRelConverter (which handles ExtTableFunction column mappings)
  // to be used wherever PlannerImpl creates a SqlToRelConverter.
  @Override
  protected SqlToRelConverter createSqlToRelConverter(
          RelOptTable.ViewExpander viewExpander,
          SqlValidator validator,
          Prepare.CatalogReader catalogReader,
          RelOptCluster cluster,
          SqlRexConvertletTable convertletTable,
          SqlToRelConverter.Config config) {
    return new HeavyDBSqlToRelConverter(
        viewExpander, validator, catalogReader, cluster, convertletTable, config);
  }

  @Override
  public RelRoot rel(SqlNode sql) {
    return super.rel(sql);
  }

  public RelRoot getRelRoot(SqlNode sqlNode) {
    return super.rel(sqlNode);
  }

  public RelNode optimizeRATree(RelNode rootNode,
          boolean viewOptimizationEnabled,
          boolean foundView,
          boolean enableExperimentalQueryRewrites) {
    HepProgramBuilder firstOptPhaseProgram = HepProgram.builder();
    firstOptPhaseProgram.addRuleInstance(CoreRules.AGGREGATE_MERGE)
            .addRuleInstance(
                    new OuterJoinOptViaNullRejectionRule(RelFactories.LOGICAL_BUILDER));
    if (enableExperimentalQueryRewrites) {
      firstOptPhaseProgram.addRuleInstance(HeavyDBNotInToAntiJoinRule.INSTANCE);
    }
    firstOptPhaseProgram
            .addRuleInstance(CoreRules.AGGREGATE_UNION_TRANSPOSE)
            .addRuleInstance(CoreRules.JOIN_PUSH_EXPRESSIONS);
    if (enableExperimentalQueryRewrites) {
      addExperimentalFirstPhaseRules(firstOptPhaseProgram);
    }
    if (!viewOptimizationEnabled) {
      firstOptPhaseProgram.addRuleInstance(CoreRules.FILTER_PROJECT_TRANSPOSE)
              .addRuleInstance(
                      FilterTableFunctionMultiInputTransposeRule.Config.DEFAULT.toRule())
              .addRuleInstance(CoreRules.FILTER_PROJECT_TRANSPOSE);
    } else {
      if (foundView) {
        firstOptPhaseProgram.addRuleInstance(
                CoreRules.JOIN_PROJECT_BOTH_TRANSPOSE_INCLUDE_OUTER);
        firstOptPhaseProgram.addRuleInstance(CoreRules.FILTER_MERGE);
      }
      firstOptPhaseProgram.addRuleInstance(CoreRules.FILTER_PROJECT_TRANSPOSE);
      firstOptPhaseProgram.addRuleInstance(
              FilterTableFunctionMultiInputTransposeRule.Config.DEFAULT.toRule());
      firstOptPhaseProgram.addRuleInstance(CoreRules.FILTER_PROJECT_TRANSPOSE);
      if (foundView) {
        firstOptPhaseProgram.addRuleInstance(CoreRules.PROJECT_MERGE);
        firstOptPhaseProgram.addRuleInstance(ProjectProjectRemoveRule.INSTANCE);
      }
    }
    HepProgram firstOptPhase = firstOptPhaseProgram.build();
    HepPlanner firstPlanner = HeavyDBPlanner.getHepPlanner(firstOptPhase, true);
    firstPlanner.setRoot(rootNode);
    final RelNode firstOptimizedPlanRoot = firstPlanner.findBestExp();
    final RelNode optimizedPlanRoot = enableExperimentalQueryRewrites
            ? cleanupUnsupportedMultiJoins(firstOptimizedPlanRoot)
            : firstOptimizedPlanRoot;

    boolean hasRLSFilter = null != restrictions && !restrictions.isEmpty();
    boolean needsSecondOptPhase = hasRLSFilter || !filterPushDownInfo.isEmpty();
    if (needsSecondOptPhase) {
      HepProgramBuilder secondOptPhaseProgram = HepProgram.builder();
      if (hasRLSFilter) {
        final InjectFilterRule injectFilterRule =
                InjectFilterRule.Config.DEFAULT.toRule(restrictions);
        secondOptPhaseProgram.addRuleInstance(injectFilterRule);
      }
      if (!filterPushDownInfo.isEmpty()) {
        final DynamicFilterJoinRule dynamicFilterJoinRule =
                new DynamicFilterJoinRule(true,
                        RelFactories.LOGICAL_BUILDER,
                        FilterJoinRule.TRUE_PREDICATE,
                        filterPushDownInfo);
        secondOptPhaseProgram.addRuleInstance(dynamicFilterJoinRule);
      }

      HepProgram secondOptPhase = secondOptPhaseProgram.build();
      HepPlanner secondPlanner = HeavyDBPlanner.getHepPlanner(secondOptPhase, true);
      secondPlanner.setRoot(optimizedPlanRoot);
      final RelNode secondOptimizedPlanRoot = secondPlanner.findBestExp();
      if (!filterPushDownInfo.isEmpty()) {
        filterPushDownInfo.clear();
      }
      return secondOptimizedPlanRoot;
    } else {
      return optimizedPlanRoot;
    }
  }

  private static void addExperimentalFirstPhaseRules(HepProgramBuilder program) {
    program.addRuleInstance(CoreRules.FILTER_INTO_JOIN)
            .addRuleInstance(CoreRules.JOIN_TO_MULTI_JOIN)
            .addRuleInstance(CoreRules.FILTER_MULTI_JOIN_MERGE)
            .addRuleInstance(CoreRules.PROJECT_MULTI_JOIN_MERGE)
            .addRuleInstance(HeavyDBDifferentValueAggregateJoinRule.INSTANCE)
            .addRuleInstance(HeavyDBDifferentValueAggregateJoinRule.MULTI_JOIN_INSTANCE)
            .addRuleInstance(HeavyDBPairedDifferentValueStatsRule.INSTANCE)
            .addRuleInstance(HeavyDBPairedDifferentValueStatsRule.MULTI_JOIN_INSTANCE)
            .addRuleInstance(HeavyDBPairedDifferentValueStatsRule.PROJECT_JOIN_INSTANCE)
            .addRuleInstance(
                    HeavyDBPairedDifferentValueStatsRule.PROJECT_MULTI_JOIN_INSTANCE)
            .addRuleInstance(
                    HeavyDBPairedDifferentValueStatsRule
                            .MERGED_FILTER_MULTI_JOIN_INSTANCE)
            .addRuleInstance(HeavyDBConnectedJoinOptimizeRule.INSTANCE)
            .addRuleInstance(HeavyDBAggregateJoinReductionRule.INSTANCE)
            .addRuleInstance(HeavyDBKeyPreservingAggregateRule.INSTANCE)
            .addRuleInstance(HeavyDBExistenceCountToGroupByRule.INSTANCE)
            .addRuleInstance(HeavyDBDifferentValueAggregateJoinRule.INSTANCE)
            .addRuleInstance(HeavyDBDifferentValueAggregateJoinRule.MULTI_JOIN_INSTANCE)
            .addRuleInstance(HeavyDBPairedDifferentValueStatsRule.INSTANCE)
            .addRuleInstance(HeavyDBPairedDifferentValueStatsRule.MULTI_JOIN_INSTANCE)
            .addRuleInstance(HeavyDBPairedDifferentValueStatsRule.PROJECT_JOIN_INSTANCE)
            .addRuleInstance(
                    HeavyDBPairedDifferentValueStatsRule.PROJECT_MULTI_JOIN_INSTANCE)
            .addRuleInstance(
                    HeavyDBPairedDifferentValueStatsRule
                            .MERGED_FILTER_MULTI_JOIN_INSTANCE)
            .addRuleInstance(HeavyDBLeftJoinAntiSemiJoinRule.PROJECT_INSTANCE)
            .addRuleInstance(HeavyDBLeftJoinAntiSemiJoinRule.INSTANCE)
            .addRuleInstance(HeavyDBAggregateJoinReductionRule.INSTANCE)
            .addRuleInstance(CoreRules.FILTER_INTO_JOIN)
            .addRuleInstance(CoreRules.JOIN_TO_MULTI_JOIN)
            .addRuleInstance(CoreRules.FILTER_MULTI_JOIN_MERGE)
            .addRuleInstance(CoreRules.PROJECT_MULTI_JOIN_MERGE)
            .addRuleInstance(HeavyDBConnectedJoinOptimizeRule.INSTANCE)
            .addRuleInstance(HeavyDBJoinFilterSplitRule.INSTANCE);
  }

  private RelNode cleanupUnsupportedMultiJoins(RelNode rootNode) {
    final HepProgram cleanupProgram =
            HepProgram.builder()
                    .addRuleInstance(HeavyDBDifferentValueAggregateJoinRule.INSTANCE)
                    .addRuleInstance(
                            HeavyDBDifferentValueAggregateJoinRule.MULTI_JOIN_INSTANCE)
                    .addRuleInstance(HeavyDBPairedDifferentValueStatsRule.INSTANCE)
                    .addRuleInstance(
                            HeavyDBPairedDifferentValueStatsRule.MULTI_JOIN_INSTANCE)
                    .addRuleInstance(
                            HeavyDBPairedDifferentValueStatsRule.PROJECT_JOIN_INSTANCE)
                    .addRuleInstance(
                            HeavyDBPairedDifferentValueStatsRule
                                    .PROJECT_MULTI_JOIN_INSTANCE)
                    .addRuleInstance(
                            HeavyDBPairedDifferentValueStatsRule
                                    .MERGED_FILTER_MULTI_JOIN_INSTANCE)
                    .addRuleInstance(HeavyDBOuterJoinMultiJoinDecomposeRule.INSTANCE)
                    .addRuleInstance(CoreRules.FILTER_INTO_JOIN)
                    .addRuleInstance(CoreRules.FILTER_PROJECT_TRANSPOSE)
                    .addRuleInstance(CoreRules.FILTER_INTO_JOIN)
                    .addRuleInstance(CoreRules.JOIN_TO_MULTI_JOIN)
                    .addRuleInstance(CoreRules.FILTER_MULTI_JOIN_MERGE)
                    .addRuleInstance(CoreRules.PROJECT_MULTI_JOIN_MERGE)
                    .addRuleInstance(HeavyDBDifferentValueAggregateJoinRule.INSTANCE)
                    .addRuleInstance(
                            HeavyDBDifferentValueAggregateJoinRule.MULTI_JOIN_INSTANCE)
                    .addRuleInstance(HeavyDBPairedDifferentValueStatsRule.INSTANCE)
                    .addRuleInstance(
                            HeavyDBPairedDifferentValueStatsRule.MULTI_JOIN_INSTANCE)
                    .addRuleInstance(
                            HeavyDBPairedDifferentValueStatsRule.PROJECT_JOIN_INSTANCE)
                    .addRuleInstance(
                            HeavyDBPairedDifferentValueStatsRule
                                    .PROJECT_MULTI_JOIN_INSTANCE)
                    .addRuleInstance(
                            HeavyDBPairedDifferentValueStatsRule
                                    .MERGED_FILTER_MULTI_JOIN_INSTANCE)
                    .addRuleInstance(HeavyDBConnectedJoinOptimizeRule.INSTANCE)
                    .addRuleInstance(HeavyDBOuterJoinMultiJoinDecomposeRule.INSTANCE)
                    .addRuleInstance(CoreRules.PROJECT_MERGE)
                    .addRuleInstance(HeavyDBDistinctAggregateJoinPruneRule.INSTANCE)
                    .addRuleInstance(HeavyDBAggregateJoinReductionRule.INSTANCE)
                    .addRuleInstance(HeavyDBKeyPreservingAggregateRule.INSTANCE)
                    .addRuleInstance(HeavyDBExistenceCountToGroupByRule.INSTANCE)
                    .addRuleInstance(HeavyDBDifferentValueAggregateJoinRule.INSTANCE)
                    .addRuleInstance(
                            HeavyDBDifferentValueAggregateJoinRule.MULTI_JOIN_INSTANCE)
                    .addRuleInstance(HeavyDBPairedDifferentValueStatsRule.INSTANCE)
                    .addRuleInstance(
                            HeavyDBPairedDifferentValueStatsRule.MULTI_JOIN_INSTANCE)
                    .addRuleInstance(
                            HeavyDBPairedDifferentValueStatsRule.PROJECT_JOIN_INSTANCE)
                    .addRuleInstance(
                            HeavyDBPairedDifferentValueStatsRule
                                    .PROJECT_MULTI_JOIN_INSTANCE)
                    .addRuleInstance(
                            HeavyDBPairedDifferentValueStatsRule
                                    .MERGED_FILTER_MULTI_JOIN_INSTANCE)
                    .addRuleInstance(HeavyDBLeftJoinAntiSemiJoinRule.PROJECT_INSTANCE)
                    .addRuleInstance(HeavyDBLeftJoinAntiSemiJoinRule.INSTANCE)
                    .addRuleInstance(HeavyDBAggregateJoinReductionRule.INSTANCE)
                    .addRuleInstance(CoreRules.FILTER_INTO_JOIN)
                    .addRuleInstance(HeavyDBLargeFilteredTableJoinRule.INSTANCE)
                    .addRuleInstance(HeavyDBNotInToAntiJoinRule.INSTANCE)
                    .addRuleInstance(CoreRules.JOIN_TO_MULTI_JOIN)
                    .addRuleInstance(CoreRules.FILTER_MULTI_JOIN_MERGE)
                    .addRuleInstance(CoreRules.PROJECT_MULTI_JOIN_MERGE)
                    .addRuleInstance(HeavyDBConnectedJoinOptimizeRule.INSTANCE)
                    .addRuleInstance(HeavyDBDifferentValueAggregateJoinRule.INSTANCE)
                    .addRuleInstance(
                            HeavyDBDifferentValueAggregateJoinRule.MULTI_JOIN_INSTANCE)
                    .addRuleInstance(HeavyDBPairedDifferentValueStatsRule.INSTANCE)
                    .addRuleInstance(
                            HeavyDBPairedDifferentValueStatsRule.MULTI_JOIN_INSTANCE)
                    .addRuleInstance(
                            HeavyDBPairedDifferentValueStatsRule.PROJECT_JOIN_INSTANCE)
                    .addRuleInstance(
                            HeavyDBPairedDifferentValueStatsRule
                                    .PROJECT_MULTI_JOIN_INSTANCE)
                    .addRuleInstance(
                            HeavyDBPairedDifferentValueStatsRule
                                    .MERGED_FILTER_MULTI_JOIN_INSTANCE)
                    .addRuleInstance(HeavyDBConnectedJoinOptimizeRule.INSTANCE)
                    .addRuleInstance(HeavyDBRedundantSemiJoinPruneRule.INSTANCE)
                    .addRuleInstance(HeavyDBJoinTreeKeysetReductionRule.PROJECT_INSTANCE)
                    .addRuleInstance(HeavyDBJoinTreeKeysetReductionRule.INSTANCE)
                    .addRuleInstance(HeavyDBRedundantSemiJoinPruneRule.INSTANCE)
                    .addRuleInstance(HeavyDBOuterJoinMultiJoinDecomposeRule.INSTANCE)
                    .addRuleInstance(HeavyDBJoinFilterSplitRule.INSTANCE)
                    .addRuleInstance(HeavyDBLeftJoinAntiSemiJoinRule.PROJECT_INSTANCE)
                    .addRuleInstance(HeavyDBLeftJoinAntiSemiJoinRule.INSTANCE)
                    .addRuleInstance(CoreRules.PROJECT_MERGE)
                    .addRuleInstance(HeavyDBLeftJoinCountDistributionRule.INSTANCE)
                    .addRuleInstance(HeavyDBSingleCountDistinctToGroupByRule.INSTANCE)
                    .addRuleInstance(HeavyDBAggregateOuterJoinStrengthReductionRule.INSTANCE)
                    .build();
    HepPlanner cleanupPlanner = HeavyDBPlanner.getHepPlanner(cleanupProgram, true);
    cleanupPlanner.setRoot(rootNode);
    final RelNode cleanedRoot = cleanupPlanner.findBestExp();

    final HepProgram aggregateFinalizeProgram =
            HepProgram.builder()
                    .addRuleInstance(HeavyDBKeyPreservingAggregateRule.INSTANCE)
                    .addRuleInstance(HeavyDBAggregateJoinReductionRule.INSTANCE)
                    .addRuleInstance(CoreRules.FILTER_INTO_JOIN)
                    .addRuleInstance(HeavyDBLargeFilteredTableJoinRule.BUILD_SIDE_INSTANCE)
                    .addRuleInstance(CoreRules.PROJECT_MERGE)
                    .addRuleInstance(HeavyDBAggregateJoinPayloadDeferralRule.PROJECT_INSTANCE)
                    .addRuleInstance(HeavyDBAggregateJoinPayloadDeferralRule.INSTANCE)
                    .addRuleInstance(HeavyDBAggregateJoinPayloadKeysetRule.INSTANCE)
                    .addRuleInstance(HeavyDBAggregateJoinWindowExtremaRule.PROJECT_INSTANCE)
                    .addRuleInstance(HeavyDBAggregateJoinWindowExtremaRule.INSTANCE)
                    .addRuleInstance(HeavyDBScalarExtremaJoinToTopNRule.INSTANCE)
                    .addRuleInstance(HeavyDBFilteredUniqueKeysetJoinRule.INSTANCE)
                    .build();
    HepPlanner aggregateFinalizePlanner =
            HeavyDBPlanner.getHepPlanner(aggregateFinalizeProgram, true);
    aggregateFinalizePlanner.setRoot(cleanedRoot);
    return aggregateFinalizePlanner.findBestExp();
  }

  private RelRoot applyInjectFilterRule(RelRoot root, List<Restriction> restrictions) {
    // TODO consider doing these rules in one preplan pass

    final InjectFilterRule injectFilterRule =
            InjectFilterRule.Config.DEFAULT.toRule(restrictions);

    final HepProgram program =
            HepProgram.builder().addRuleInstance(injectFilterRule).build();
    HepPlanner prePlanner = HeavyDBPlanner.getHepPlanner(program, false);
    prePlanner.setRoot(root.rel);
    final RelNode rootRelNode = prePlanner.findBestExp();
    return root.withRel(rootRelNode);
  }

  private RelRoot applyFilterPushdown(RelRoot root) {
    if (filterPushDownInfo.isEmpty()) {
      return root;
    }
    final DynamicFilterJoinRule dynamicFilterJoinRule = new DynamicFilterJoinRule(true,
            RelFactories.LOGICAL_BUILDER,
            FilterJoinRule.TRUE_PREDICATE,
            filterPushDownInfo);
    final HepProgram program =
            HepProgram.builder().addRuleInstance(dynamicFilterJoinRule).build();
    HepPlanner prePlanner = HeavyDBPlanner.getHepPlanner(program, false);
    prePlanner.setRoot(root.rel);
    final RelNode rootRelNode = prePlanner.findBestExp();
    filterPushDownInfo.clear();
    return root.withRel(rootRelNode);
  }

  private RelRoot applyOptimizationsRules(RelRoot root, ImmutableSet<RelOptRule> rules) {
    HepProgramBuilder programBuilder = new HepProgramBuilder();
    for (RelOptRule rule : rules) {
      programBuilder.addRuleInstance(rule);
    }
    HepPlanner hepPlanner = HeavyDBPlanner.getHepPlanner(programBuilder.build(), false);
    hepPlanner.setRoot(root.rel);
    return root.withRel(hepPlanner.findBestExp());
  }

  public RelRoot buildRATreeAndPerformQueryOptimization(
          String query, HeavyDBSchema schema) throws IOException {
    ready();
    RexBuilder builder = new HeavyDBRexBuilder(getTypeFactory());
    RelOptCluster cluster = RelOptCluster.create(new VolcanoPlanner(), builder);
    cluster.getPlanner().setExecutor(new HeavyDBRexExecutor(DataContexts.EMPTY));
    CalciteCatalogReader catalogReader = createCatalogReader();
    HeavyDBRelJsonReader reader =
            new HeavyDBRelJsonReader(cluster, catalogReader, schema);

    RelRoot relR = RelRoot.of(reader.read(query), SqlKind.SELECT);

    if (restrictions != null) {
      relR = applyInjectFilterRule(relR, restrictions);
    }
    QueryOptimizationRules outerJoinOptRule =
            new OuterJoinOptViaNullRejectionRule(RelFactories.LOGICAL_BUILDER);
    relR = applyOptimizationsRules(
            relR, ImmutableSet.of(CoreRules.AGGREGATE_MERGE, outerJoinOptRule));
    relR = applyFilterPushdown(relR);
    relR = applyOptimizationsRules(relR,
            ImmutableSet.of(CoreRules.JOIN_PROJECT_BOTH_TRANSPOSE_INCLUDE_OUTER,
                    CoreRules.FILTER_REDUCE_EXPRESSIONS,
                    ProjectProjectRemoveRule.INSTANCE,
                    CoreRules.PROJECT_FILTER_TRANSPOSE));
    relR = applyOptimizationsRules(relR, ImmutableSet.of(CoreRules.PROJECT_MERGE));
    relR = applyOptimizationsRules(relR,
            ImmutableSet.of(
                    CoreRules.FILTER_PROJECT_TRANSPOSE, CoreRules.PROJECT_REMOVE));
    return RelRoot.of(relR.project(), relR.kind);
  }

  public void setFilterPushDownInfo(
          final List<HeavyDBParserOptions.FilterPushDownInfo> filterPushDownInfo) {
    this.filterPushDownInfo = filterPushDownInfo;
  }

  public void setRestrictions(List<Restriction> restrictions) {
    this.restrictions = restrictions;
  }
}

// End HeavyDBPlanner.java
