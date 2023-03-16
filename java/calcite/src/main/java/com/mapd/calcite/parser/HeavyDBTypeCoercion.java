package com.mapd.calcite.parser;

import static com.mapd.parser.server.ExtensionFunction.*;

import com.mapd.calcite.parser.HeavyDBSqlOperatorTable.ExtTableFunction;

import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeFactoryImpl;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rel.type.RelDataTypeFieldImpl;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlCallBinding;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlSelect;
import org.apache.calcite.sql.type.ArraySqlType;
import org.apache.calcite.sql.type.IntervalSqlType;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.type.SqlTypeUtil;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.sql.validate.SqlValidatorScope;
import org.apache.calcite.sql.validate.implicit.TypeCoercionImpl;

import java.util.ArrayList;
import java.util.List;

public class HeavyDBTypeCoercion extends TypeCoercionImpl {
  public HeavyDBTypeCoercion(RelDataTypeFactory typeFactory, SqlValidator validator) {
    super(typeFactory, validator);
  }

  /**
   * Calculates a type coercion score for a given call to a User-Defined Table Function,
   * assuming the call is bound to overload @udtf. This is used to perform overload
   * resolution and type checking while considering implicit type coercions are performed
   * (automatic casting of input scalar/columnar arguments).
   */
  public int calculateTypeCoercionScore(
          SqlCallBinding callBinding, ExtTableFunction udtf) {
    final List<ExtArgumentType> paramTypes = udtf.getArgTypes();
    SqlCall permutedCall = callBinding.permutedCall();
    assert paramTypes != null;
    int score = 0;
    for (int i = 0; i < permutedCall.operandCount(); i++) {
      SqlNode operand = permutedCall.operand(i);

      // DEFAULT operands should always type-check without any casting (they will be
      // filled in later during sql2rel translation)
      if (operand.getKind() == SqlKind.DEFAULT) {
        continue;
      }

      RelDataType actualRelType = validator.deriveType(callBinding.getScope(), operand);
      RelDataType formalRelType = toRelDataType(paramTypes.get(i), factory);

      if (actualRelType.getSqlTypeName() == SqlTypeName.CURSOR) {
        SqlCall cursorCall = (SqlCall) operand;
        int cursorScore = calculateScoreForCursorOperand(
                cursorCall.operand(0), i, callBinding.getScope(), udtf);
        if (cursorScore < 0) {
          return -1;
        }
        score += cursorScore;
      } else if (actualRelType != formalRelType) {
        if (SqlTypeUtil.isInterval(actualRelType)
                && SqlTypeUtil.isInterval(formalRelType)) {
          // intervals only need to match on whether they're YearMonth or Day
          IntervalSqlType actualInterval = (IntervalSqlType) actualRelType;
          IntervalSqlType formalInterval = (IntervalSqlType) formalRelType;
          if (actualInterval.getIntervalQualifier().isYearMonth()
                  == formalInterval.getIntervalQualifier().isYearMonth()) {
            continue;
          } else {
            return -1;
          }
        }
        RelDataType widerType = getWiderTypeForTwo(actualRelType, formalRelType, false);
        if (widerType == null) {
          return -1;
        } else if (!SqlTypeUtil.isTimestamp(widerType)
                && SqlTypeUtil.sameNamedType(formalRelType, actualRelType)) {
          // TIMESTAMPs of different precision need coercion, but other structurally equal
          // types do not (i.e VARCHAR(1) and VARCHAR(10))
          continue;
        } else if (actualRelType == widerType) {
          return -1;
        } else if (widerType != actualRelType) {
          score += getScoreForTypes(widerType, actualRelType, false);
        }
      }
    }
    return score;
  }

  /**
   * Calculates a type coercion score for a single CURSOR() operand. CURSORS need special
   * logic because their underlying types are composite types of one or multiple columns.
   */
  private int calculateScoreForCursorOperand(SqlNode cursorOperand,
          int index,
          SqlValidatorScope scope,
          ExtTableFunction udtf) {
    int score = 0;

    String formalOperandName = udtf.getExtendedParamNames().get(index);
    List<ExtArgumentType> formalFieldTypes =
            udtf.getCursorFieldTypes().get(formalOperandName);

    if (formalFieldTypes == null || formalFieldTypes.size() == 0) {
      System.out.println(
              "Warning: UDTF has no CURSOR field subtype data. Proceeding assuming CURSOR typechecks.");
      // penalize implementations without subtype data, to favor ones that have better
      // type matches
      return 1000;
    }

    switch (cursorOperand.getKind()) {
      case SELECT: {
        SqlSelect selectNode = (SqlSelect) cursorOperand;

        int iFormal = 0, iActual = 0;
        for (; iActual < selectNode.getSelectList().size()
                && iFormal < formalFieldTypes.size();
                iActual++, iFormal++) {
          SqlNode selectOperand = selectNode.getSelectList().get(iActual);
          ExtArgumentType extType = formalFieldTypes.get(iFormal);
          RelDataType formalRelType = toRelDataType(extType, factory);
          RelDataType actualRelType = factory.createTypeWithNullability(
                  validator.deriveType(scope, selectOperand), true);

          if (formalRelType.getSqlTypeName() == SqlTypeName.COLUMN_LIST) {
            ExtArgumentType colListSubtype = getValueType(extType);
            RelDataType formalSubtype = toRelDataType(colListSubtype, factory);

            if (isArrayType(colListSubtype)
                    && actualRelType.getSqlTypeName() == SqlTypeName.ARRAY) {
              ArraySqlType formalArrayType = (ArraySqlType) formalSubtype;
              ArraySqlType actualArrayType = (ArraySqlType) actualRelType;
              if (!SqlTypeUtil.sameNamedType(formalArrayType.getComponentType(),
                          actualArrayType.getComponentType())) {
                // Arrays are not castable, so arrays whose underlying types don't match
                // can fail early
                return -1;
              }
            }

            RelDataType widerType =
                    getWiderTypeForTwo(actualRelType, formalSubtype, false);
            if (!SqlTypeUtil.sameNamedType(actualRelType, formalSubtype)) {
              if (widerType == null || widerType == actualRelType) {
                // no common type, or actual type is wider than formal
                return -1;
              }
            }

            int colListSize = 0;
            int numFormalArgumentsLeft = (formalFieldTypes.size() - 1) - iFormal;
            int maxColListSize =
                    selectNode.getSelectList().size() - numFormalArgumentsLeft - iActual;
            while (colListSize < maxColListSize) {
              SqlNode curOperand = selectNode.getSelectList().get(iActual + colListSize);
              actualRelType = scope.getValidator().deriveType(scope, curOperand);
              widerType = getWiderTypeForTwo(formalSubtype, actualRelType, false);
              if (!SqlTypeUtil.sameNamedType(actualRelType, formalSubtype)) {
                if (widerType == null
                        || !SqlTypeUtil.sameNamedType(widerType, formalSubtype)) {
                  // no common type, or actual type is wider than formal
                  break;
                } else if (widerType != formalSubtype) {
                  // formal subtype is narrower than widerType, we do not support
                  // downcasting
                  break;
                } else {
                  score += getScoreForTypes(widerType, actualRelType, true);
                }
              }
              colListSize++;
            }
            iActual += colListSize - 1;
          } else if (actualRelType != formalRelType) {
            RelDataType widerType =
                    getWiderTypeForTwo(formalRelType, actualRelType, false);
            if (widerType == null) {
              // no common wider type
              return -1;
            } else if (!SqlTypeUtil.isTimestamp(widerType)
                    && SqlTypeUtil.sameNamedType(formalRelType, actualRelType)) {
              // TIMESTAMPs of different precision need coercion, but other structurally
              // equal types do not (i.e VARCHAR(1) and VARCHAR(10))
              continue;
            } else if (actualRelType == widerType
                    || !SqlTypeUtil.sameNamedType(widerType, formalRelType)) {
              // formal type is narrower than widerType or the provided actualType, we do
              // not support downcasting
              return -1;
            } else {
              score += getScoreForTypes(widerType, actualRelType, true);
            }
          }
        }

        if (iActual < selectNode.getSelectList().size()) {
          return -1;
        }
        return score;
      }
      default: {
        System.out.println("Unsupported subquery kind in UDTF CURSOR input argument: "
                + cursorOperand.getKind());
        return -1;
      }
    }
  }

  /**
   * We overload this to add a few additional useful rules that Calcite does not implement
   * by default: TIMESTAMP() of lower precision should be castable to higher precision
   * DOUBLE should be the widest type for {FLOAT, REAL, DOUBLE}
   */
  public RelDataType getWiderTypeForTwo(
          RelDataType type1, RelDataType type2, boolean stringPromotion) {
    RelDataType returnType = super.getWiderTypeForTwo(type1, type2, stringPromotion);
    if (SqlTypeUtil.isTimestamp(type1) && SqlTypeUtil.isTimestamp(type2)) {
      returnType = (type1.getPrecision() > type2.getPrecision()) ? type1 : type2;
    } else if ((SqlTypeUtil.isDouble(type1) || SqlTypeUtil.isDouble(type2))
            && (SqlTypeUtil.isApproximateNumeric(type1)
                    && SqlTypeUtil.isApproximateNumeric(type2))) {
      returnType = factory.createTypeWithNullability(
              factory.createSqlType(SqlTypeName.DOUBLE), true);
    }
    return returnType;
  }

  /**
   * After a call has been bound to the lowest cost overload, this method performs any
   * necessary coercion of input scalar/columnar arguments in the call.
   */
  public boolean extTableFunctionTypeCoercion(
          SqlCallBinding callBinding, ExtTableFunction udtf) {
    boolean coerced = false;
    final List<ExtArgumentType> paramTypes = udtf.getArgTypes();
    SqlCall permutedCall = callBinding.permutedCall();
    for (int i = 0; i < permutedCall.operandCount(); i++) {
      SqlNode operand = permutedCall.operand(i);
      if (operand.getKind() == SqlKind.DEFAULT) {
        // DEFAULT operands don't need to be coerced, they will be filled in by
        // appropriately typed constants later
        continue;
      }

      RelDataType actualRelType = validator.deriveType(callBinding.getScope(), operand);

      if (actualRelType.getSqlTypeName() == SqlTypeName.CURSOR) {
        SqlCall cursorCall = (SqlCall) operand;
        coerceCursorType(
                callBinding.getScope(), permutedCall, i, cursorCall.operand(0), udtf);
      }

      RelDataType formalRelType = toRelDataType(paramTypes.get(i), factory);
      if (actualRelType != formalRelType) {
        if (SqlTypeUtil.isInterval(actualRelType)
                && SqlTypeUtil.isInterval(formalRelType)) {
          IntervalSqlType actualInterval = (IntervalSqlType) actualRelType;
          IntervalSqlType formalInterval = (IntervalSqlType) formalRelType;
          if (actualInterval.getIntervalQualifier().isYearMonth()
                  == formalInterval.getIntervalQualifier().isYearMonth()) {
            continue;
          }
        }
        RelDataType widerType = getWiderTypeForTwo(actualRelType, formalRelType, false);
        if (!SqlTypeUtil.isTimestamp(widerType)
                && SqlTypeUtil.sameNamedType(formalRelType, actualRelType)) {
          // TIMESTAMPs of different precision need coercion, but other structurally equal
          // types do not (i.e VARCHAR(1) and VARCHAR(10))
          continue;
        }
        coerced = coerceOperandType(callBinding.getScope(), permutedCall, i, widerType)
                || coerced;
      }
    }
    return coerced;
  }

  private void coerceCursorType(SqlValidatorScope scope,
          SqlCall call,
          int index,
          SqlNode cursorOperand,
          ExtTableFunction udtf) {
    String formalOperandName = udtf.getExtendedParamNames().get(index);
    List<ExtArgumentType> formalFieldTypes =
            udtf.getCursorFieldTypes().get(formalOperandName);
    if (formalFieldTypes == null || formalFieldTypes.size() == 0) {
      return;
    }

    switch (cursorOperand.getKind()) {
      case SELECT: {
        SqlSelect selectNode = (SqlSelect) cursorOperand;
        int iFormal = 0, iActual = 0;
        List<RelDataTypeField> newValidatedTypeList = new ArrayList<>();
        for (; iActual < selectNode.getSelectList().size()
                && iFormal < formalFieldTypes.size();
                iFormal++, iActual++) {
          SqlNode selectOperand = selectNode.getSelectList().get(iActual);
          ExtArgumentType extType = formalFieldTypes.get(iFormal);
          RelDataType formalRelType = toRelDataType(extType, factory);
          RelDataType actualRelType = validator.deriveType(scope, selectOperand);

          if (isColumnArrayType(extType) || isColumnListArrayType(extType)) {
            // Arrays can't be casted so don't bother trying
            updateValidatedType(newValidatedTypeList, selectNode, iActual);
            continue;
          }

          if (formalRelType.getSqlTypeName() == SqlTypeName.COLUMN_LIST) {
            ExtArgumentType colListSubtype = getValueType(extType);
            RelDataType formalSubtype = toRelDataType(colListSubtype, factory);
            RelDataType widerType =
                    getWiderTypeForTwo(actualRelType, formalSubtype, false);

            int colListSize = 0;
            int numFormalArgumentsLeft = (formalFieldTypes.size() - 1) - iFormal;
            int maxColListSize =
                    selectNode.getSelectList().size() - numFormalArgumentsLeft - iActual;
            while (colListSize < maxColListSize) {
              SqlNode curOperand = selectNode.getSelectList().get(iActual + colListSize);
              actualRelType = scope.getValidator().deriveType(scope, curOperand);
              widerType = getWiderTypeForTwo(formalSubtype, actualRelType, false);
              if (!SqlTypeUtil.sameNamedType(actualRelType, formalSubtype)) {
                if (widerType == null) {
                  break;
                } else if (actualRelType != widerType) {
                  coerceColumnType(scope,
                          selectNode.getSelectList(),
                          iActual + colListSize,
                          widerType);
                }
              }
              updateValidatedType(
                      newValidatedTypeList, selectNode, iActual + colListSize);
              colListSize++;
            }
            iActual += colListSize - 1;
          } else if (actualRelType != formalRelType) {
            RelDataType widerType =
                    getWiderTypeForTwo(formalRelType, actualRelType, false);
            if (!SqlTypeUtil.isTimestamp(widerType)
                    && SqlTypeUtil.sameNamedType(actualRelType, formalRelType)) {
              updateValidatedType(newValidatedTypeList, selectNode, iActual);
              continue;
            }
            if (widerType != actualRelType) {
              coerceColumnType(scope, selectNode.getSelectList(), iActual, widerType);
            }
            updateValidatedType(newValidatedTypeList, selectNode, iActual);
          } else {
            // keep old validated type for argument that was not coerced
            updateValidatedType(newValidatedTypeList, selectNode, iActual);
          }
        }
        RelDataType newCursorStructType = factory.createStructType(newValidatedTypeList);
        RelDataType newCursorType = factory.createTypeWithNullability(newCursorStructType,
                validator.getValidatedNodeType(selectNode).isNullable());
        validator.setValidatedNodeType(selectNode, newCursorType);
        break;
      }
      default: {
        return;
      }
    }
  }

  /**
   * We overload this specifically to REMOVE a rule Calcite uses: Calcite will not cast
   * types in the NUMERIC type family across each other. Therefore, with Calcite's default
   * rules, we would not cast INTEGER columns to BIGINT, or FLOAT to DOUBLE.
   */
  @Override
  protected boolean needToCast(
          SqlValidatorScope scope, SqlNode node, RelDataType toType) {
    RelDataType fromType = validator.deriveType(scope, node);
    // This depends on the fact that type validate happens before coercion.
    // We do not have inferred type for some node, i.e. LOCALTIME.
    if (fromType == null) {
      return false;
    }

    // This prevents that we cast a JavaType to normal RelDataType.
    if (fromType instanceof RelDataTypeFactoryImpl.JavaType
            && toType.getSqlTypeName() == fromType.getSqlTypeName()) {
      return false;
    }

    // Do not make a cast when we don't know specific type (ANY) of the origin node.
    if (toType.getSqlTypeName() == SqlTypeName.ANY
            || fromType.getSqlTypeName() == SqlTypeName.ANY) {
      return false;
    }

    // No need to cast between char and varchar.
    if (SqlTypeUtil.isCharacter(toType) && SqlTypeUtil.isCharacter(fromType)) {
      return false;
    }

    // Implicit type coercion does not handle nullability.
    if (SqlTypeUtil.equalSansNullability(factory, fromType, toType)) {
      return false;
    }
    // Should keep sync with rules in SqlTypeCoercionRule.
    assert SqlTypeUtil.canCastFrom(toType, fromType, true);
    return true;
  }

  /**
   * Updates validated type for CURSOR SqlNodes that had their underlying types coerced.
   */
  private void updateValidatedType(
          List<RelDataTypeField> typeList, SqlSelect selectNode, int operandIndex) {
    SqlNode operand = selectNode.getSelectList().get(operandIndex);
    RelDataType newType = validator.getValidatedNodeType(operand);
    if (operand instanceof SqlCall) {
      SqlCall asCall = (SqlCall) operand;
      if (asCall.getOperator().kind == SqlKind.AS) {
        newType = validator.getValidatedNodeType(asCall.operand(0));
      }
    }
    RelDataTypeField oldTypeField =
            validator.getValidatedNodeType(selectNode).getFieldList().get(operandIndex);
    RelDataTypeField newTypeField = new RelDataTypeFieldImpl(
            oldTypeField.getName(), oldTypeField.getIndex(), newType);
    typeList.add(newTypeField);
  }

  /**
   * Returns a coercion score between an input type and a target type to be casted to.
   * Currently we consider cursor arguments as 100x more expensive than scalar arguments.
   * We also consider a 10x cost to cast from integer to floating point types, to favor
   * overloads that conserve precision.
   */
  private int getScoreForTypes(
          RelDataType targetType, RelDataType originalType, boolean isCursorArgument) {
    // we assume casting columns is 100x more expensive than casting scalars
    int baseScore = isCursorArgument ? 100 : 1;
    switch (originalType.getSqlTypeName()) {
      case TINYINT:
      case SMALLINT:
      case INTEGER:
      case BIGINT: {
        int multiplier = 1;
        if (SqlTypeUtil.isApproximateNumeric(targetType)) {
          // should favor keeping integer types over promoting to floating point
          multiplier = 10;
        } /* TODO: Re-enable cast to string types after ColumnList binding is resolved
         else if (SqlTypeUtil.inCharFamily(targetType)) {
           // promoting to char types should be a last resort
           multiplier = 1000;
         }*/
        return baseScore * multiplier;
      }
      default: {
        // promoting to char types should be a last resort
        /* TODO: Re-enable cast to string types after ColumnList binding is resolved
        int multiplier = (SqlTypeUtil.inCharFamily(targetType) ? 1000 : 1);
        */
        return baseScore /** multiplier*/;
      }
    }
  }
}