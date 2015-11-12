/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mapd.calcite.parser;

import java.util.List;
import java.util.Map;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.runtime.CalciteContextException;
import org.apache.calcite.runtime.CalciteException;
import org.apache.calcite.runtime.Resources;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlDataTypeSpec;
import org.apache.calcite.sql.SqlDelete;
import org.apache.calcite.sql.SqlDynamicParam;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlInsert;
import org.apache.calcite.sql.SqlIntervalQualifier;
import org.apache.calcite.sql.SqlLiteral;
import org.apache.calcite.sql.SqlMerge;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlNodeList;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.SqlSelect;
import org.apache.calcite.sql.SqlUpdate;
import org.apache.calcite.sql.SqlWindow;
import org.apache.calcite.sql.SqlWith;
import org.apache.calcite.sql.SqlWithItem;
import org.apache.calcite.sql.validate.SelectScope;
import org.apache.calcite.sql.validate.SqlConformance;
import org.apache.calcite.sql.validate.SqlModality;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.sql.validate.SqlValidatorCatalogReader;
import org.apache.calcite.sql.validate.SqlValidatorException;
import org.apache.calcite.sql.validate.SqlValidatorNamespace;
import org.apache.calcite.sql.validate.SqlValidatorScope;

/**
 *
 * @author michael
 */
class MapDSqlValidator implements SqlValidator {

  public MapDSqlValidator() {
  }

  @Override
  public SqlConformance getConformance() {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public SqlValidatorCatalogReader getCatalogReader() {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public SqlOperatorTable getOperatorTable() {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public SqlNode validate(SqlNode topNode) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public SqlNode validateParameterizedExpression(SqlNode topNode, Map<String,
          RelDataType> nameToTypeMap) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void validateQuery(SqlNode node, SqlValidatorScope scope) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public RelDataType getValidatedNodeType(SqlNode node) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public RelDataType getValidatedNodeTypeIfKnown(SqlNode node) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void validateIdentifier(SqlIdentifier id, SqlValidatorScope scope) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void validateLiteral(SqlLiteral literal) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void validateIntervalQualifier(SqlIntervalQualifier qualifier) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void validateInsert(SqlInsert insert) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void validateUpdate(SqlUpdate update) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void validateDelete(SqlDelete delete) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void validateMerge(SqlMerge merge) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void validateDataType(SqlDataTypeSpec dataType) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void validateDynamicParam(SqlDynamicParam dynamicParam) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void validateWindow(SqlNode windowOrId, SqlValidatorScope scope, SqlCall call) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void validateCall(SqlCall call, SqlValidatorScope scope) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void validateAggregateParams(SqlCall aggCall, SqlNode filter, SqlValidatorScope scope) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void validateColumnListParams(SqlFunction function, List<RelDataType> argTypes, List<SqlNode> operands) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public RelDataType deriveType(SqlValidatorScope scope, SqlNode operand) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public CalciteContextException newValidationError(SqlNode node, Resources.ExInst<SqlValidatorException> e) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public boolean isAggregate(SqlSelect select) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public boolean isAggregate(SqlNode selectNode) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public SqlWindow resolveWindow(SqlNode windowOrRef, SqlValidatorScope scope, boolean populateBounds) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public SqlValidatorNamespace getNamespace(SqlNode node) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public String deriveAlias(SqlNode node, int ordinal) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public SqlNodeList expandStar(SqlNodeList selectList, SqlSelect query, boolean includeSystemVars) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public SqlValidatorScope getWhereScope(SqlSelect select) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public RelDataTypeFactory getTypeFactory() {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void setValidatedNodeType(SqlNode node, RelDataType type) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void removeValidatedNodeType(SqlNode node) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public RelDataType getUnknownType() {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public SqlValidatorScope getSelectScope(SqlSelect select) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public SelectScope getRawSelectScope(SqlSelect select) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public SqlValidatorScope getFromScope(SqlSelect select) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public SqlValidatorScope getJoinScope(SqlNode node) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public SqlValidatorScope getGroupScope(SqlSelect select) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public SqlValidatorScope getHavingScope(SqlSelect select) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public SqlValidatorScope getOrderScope(SqlSelect select) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void declareCursor(SqlSelect select, SqlValidatorScope scope) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void pushFunctionCall() {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void popFunctionCall() {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public String getParentCursor(String columnListParamName) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void setIdentifierExpansion(boolean expandIdentifiers) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void setColumnReferenceExpansion(boolean expandColumnReferences) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public boolean getColumnReferenceExpansion() {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public boolean shouldExpandIdentifiers() {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void setCallRewrite(boolean rewriteCalls) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public RelDataType deriveConstructorType(SqlValidatorScope scope, SqlCall call, SqlFunction unresolvedConstructor, SqlFunction resolvedConstructor, List<RelDataType> argTypes) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public CalciteException handleUnresolvedFunction(SqlCall call, SqlFunction unresolvedFunction, List<RelDataType> argTypes, List<String> argNames) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public SqlNode expandOrderExpr(SqlSelect select, SqlNode orderExpr) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public SqlNode expand(SqlNode expr, SqlValidatorScope scope) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public boolean isSystemField(RelDataTypeField field) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public List<List<String>> getFieldOrigins(SqlNode sqlQuery) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public RelDataType getParameterRowType(SqlNode sqlQuery) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public SqlValidatorScope getOverScope(SqlNode node) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public boolean validateModality(SqlSelect select, SqlModality modality, boolean fail) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void validateWith(SqlWith with, SqlValidatorScope scope) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void validateWithItem(SqlWithItem withItem) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public SqlValidatorScope getWithScope(SqlNode withItem) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

}
