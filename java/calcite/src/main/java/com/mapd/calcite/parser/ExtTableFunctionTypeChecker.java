package com.mapd.calcite.parser;

import static com.mapd.parser.server.ExtensionFunction.*;

import static org.apache.calcite.runtime.Resources.BaseMessage;
import static org.apache.calcite.runtime.Resources.ExInst;

import com.mapd.calcite.parser.HeavyDBSqlOperatorTable.ExtTableFunction;

import org.apache.calcite.linq4j.Ord;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.runtime.CalciteException;
import org.apache.calcite.runtime.Resources;
import org.apache.calcite.sql.SqlBasicCall;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlCallBinding;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperandCountRange;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSyntax;
import org.apache.calcite.sql.SqlUtil;
import org.apache.calcite.sql.type.SqlOperandCountRanges;
import org.apache.calcite.sql.type.SqlOperandTypeChecker;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.validate.SqlNameMatchers;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.sql.validate.SqlValidatorException;
import org.apache.calcite.sql.validate.SqlValidatorScope;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ExtTableFunctionTypeChecker implements SqlOperandTypeChecker {
  final HeavyDBSqlOperatorTable opTable;

  ExtTableFunctionTypeChecker(HeavyDBSqlOperatorTable opTable) {
    this.opTable = opTable;
  }

  /*
   * This function is meant to check optionality when typechecking single
   * operators. Since we perform type checking of all possible overloads
   * of an operator, we use each UDTF's isArgumentOptional() method instead.
   */
  public boolean isOptional(int argIndex) {
    return false;
  }

  public boolean checkOperandTypes(SqlCallBinding callBinding, boolean throwOnFailure) {
    Set<ExtTableFunction> candidateOverloads = new HashSet<ExtTableFunction>(
            getOperatorOverloads(callBinding.getOperator()));

    for (SqlNode operand : callBinding.getCall().getOperandList()) {
      if (operand != null && operand.getKind() == SqlKind.ARGUMENT_ASSIGNMENT) {
        final SqlCall assignmentCall = (SqlCall) operand;
        final SqlIdentifier id = assignmentCall.operand(1);
        final String paramName = id.getSimple();
        if (!candidateOverloads.stream().anyMatch(
                    tf -> tf.getParamNames().contains(paramName))) {
          throw newExtTableFunctionNameError(callBinding, operand, id.getSimple());
        }
      }
    }

    // Remove all candidates whose number of formal args doesn't match the
    // call's number of real args (accounting for possible default args).
    candidateOverloads.removeIf(tf
            -> (callBinding.getOperandCount()
                       < (tf.getArgTypes().size() - tf.getNumOptionalArguments()))
                    || (callBinding.getOperandCount() > tf.getArgTypes().size()));

    SqlNode[] operandArray = new SqlNode[callBinding.getCall().getOperandList().size()];
    for (Ord<SqlNode> arg : Ord.zip(callBinding.getCall().getOperandList())) {
      operandArray[arg.i] = arg.e;
    }

    // Construct a candidate call binding for each overload. We need to do this because
    // type inference of operands may differ depending on which operator is used. Thus,
    // typechecking needs to be done on a candidate call-by-call basis.
    HashMap<ExtTableFunction, SqlCallBinding> candidateBindings =
            new HashMap<>(candidateOverloads.size());
    for (ExtTableFunction tf : candidateOverloads) {
      SqlBasicCall newCall = new SqlBasicCall(
              tf, operandArray, callBinding.getCall().getParserPosition());
      SqlCallBinding candidateBinding = new SqlCallBinding(
              callBinding.getValidator(), callBinding.getScope(), newCall);
      candidateBindings.put(tf, candidateBinding);
    }

    // remove candidate calls that have DEFAULT values for operands which are
    // mandatory
    candidateOverloads.removeIf(tf
            -> IntStream
                       .range(0,
                               candidateBindings.get(tf)
                                       .permutedCall()
                                       .getOperandList()
                                       .size())
                       .anyMatch(idx
                               -> candidateBindings.get(tf)
                                                       .permutedCall()
                                                       .operand(idx)
                                                       .getKind()
                                               == SqlKind.DEFAULT
                                       && !tf.isArgumentOptional(idx)));

    // Compute a typechecking score for each candidate, taking into account type promotion
    HeavyDBTypeCoercion tc =
            (HeavyDBTypeCoercion) callBinding.getValidator().getTypeCoercion();
    Map<ExtTableFunction, Integer> scoredCandidates =
            new HashMap<ExtTableFunction, Integer>();
    candidateOverloads.stream().forEach(udtf
            -> scoredCandidates.put(udtf,
                    tc.calculateTypeCoercionScore(candidateBindings.get(udtf), udtf)));

    // Use the candidate with minimum cost
    ExtTableFunction minCandidate;
    try {
      minCandidate = Collections
                             .min(scoredCandidates.entrySet()
                                             .stream()
                                             .filter(entry -> entry.getValue() >= 0)
                                             .collect(Collectors.toSet()),
                                     Map.Entry.comparingByValue())
                             .getKey();
    } catch (NoSuchElementException e) {
      // If there are no candidates left, the call is invalid.
      if (throwOnFailure) {
        throw(newExtTableFunctionSignatureError(callBinding));
      }
      return false;
    }

    if (scoredCandidates.get(minCandidate) > 0) {
      tc.extTableFunctionTypeCoercion(candidateBindings.get(minCandidate), minCandidate);
    }

    // If there are candidates left, and the current bound operator
    // is not one of them, rewrite the call to use a better binding.
    ((SqlBasicCall) callBinding.getCall()).setOperator(minCandidate);

    return true;
  }

  public List<ExtTableFunction> getOperatorOverloads(SqlOperator op) {
    List<SqlOperator> overloads = new ArrayList<>();
    opTable.lookupOperatorOverloads(op.getNameAsId(),
            SqlFunctionCategory.USER_DEFINED_TABLE_FUNCTION,
            SqlSyntax.FUNCTION,
            overloads,
            SqlNameMatchers.liberal());

    return overloads.stream()
            .filter(p -> p instanceof ExtTableFunction)
            .map(p -> (ExtTableFunction) p)
            .collect(Collectors.toList());
  }

  public SqlOperandCountRange getOperandCountRange() {
    return SqlOperandCountRanges.any();
  }

  public String getAllowedSignatures(SqlOperator op, String opName) {
    List<ExtTableFunction> overloads = getOperatorOverloads(op);
    return String.join(System.lineSeparator() + "\t",
            overloads.stream()
                    .map(tf -> tf.getExtendedSignature())
                    .collect(Collectors.toList()));
  }

  public Consistency getConsistency() {
    return Consistency.NONE;
  }

  public CalciteException newExtTableFunctionNameError(
          SqlCallBinding callBinding, SqlNode operand, String operandName) {
    return callBinding.getValidator().newValidationError(operand,
            UDTF_ERRORS.paramNameMismatch(
                    callBinding.getOperator().getName(), operandName));
  }

  public CalciteException newExtTableFunctionSignatureError(SqlCallBinding callBinding) {
    return callBinding.getValidator().newValidationError(callBinding.permutedCall(),
            UDTF_ERRORS.typeMismatch(callBinding.getOperator().getName(),
                    getCallSignature(callBinding,
                            callBinding.getValidator(),
                            callBinding.getScope()),
                    System.getProperty("line.separator") + "\t"
                            + callBinding.getOperator().getAllowedSignatures()));
  }

  // Returns a call signature with detailed CURSOR type information, to emit
  // better error messages
  public String getCallSignature(
          SqlCallBinding callBinding, SqlValidator validator, SqlValidatorScope scope) {
    List<String> signatureList = new ArrayList<>();
    for (final SqlNode operand : callBinding.permutedCall().getOperandList()) {
      final RelDataType argType = validator.deriveType(scope, operand);
      if (null == argType) {
        continue;
      } else if (argType.getSqlTypeName() == SqlTypeName.CURSOR) {
        SqlCall cursorCall = (SqlCall) operand;
        RelDataType cursorType = callBinding.getValidator().deriveType(
                callBinding.getScope(), cursorCall.operand(0));
        StringBuilder cursorTypeName = new StringBuilder();
        cursorTypeName.append("CURSOR[");
        for (int j = 0; j < cursorType.getFieldList().size(); j++) {
          if (j > 0) {
            cursorTypeName.append(",");
          }
          cursorTypeName.append(
                  cursorType.getFieldList().get(j).getType().getSqlTypeName());
        }
        cursorTypeName.append("]");
        signatureList.add(cursorTypeName.toString());
      } else {
        signatureList.add(argType.toString());
      }
    }
    return SqlUtil.getOperatorSignature(callBinding.getOperator(), signatureList);
  }

  public interface ExtTableFunctionErrors {
    @BaseMessage(
            "No candidate for User-defined Table Function ''{0}'' with input parameter named ''{1}''")
    ExInst<SqlValidatorException>
    paramNameMismatch(String udtf, String wrongParamName);

    @BaseMessage(
            "Cannot apply User-defined Table Function ''{0}'' to arguments of type {1}. Supported form(s): {2}")
    ExInst<SqlValidatorException>
    typeMismatch(String udtf, String call, String overloads);
  }

  public static final ExtTableFunctionErrors UDTF_ERRORS =
          Resources.create(ExtTableFunctionErrors.class);
}
