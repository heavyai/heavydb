/*
 *  Some cool MapD License
 */
package com.mapd.calcite.parser;

import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.validate.SqlConformance;
import org.apache.calcite.sql.validate.SqlValidatorCatalogReader;
import org.apache.calcite.sql.validate.SqlValidatorImpl;

/**
 *
 * @author michael
 */
public class MapDValidator extends SqlValidatorImpl {

  public MapDValidator(
          SqlOperatorTable opTab,
          SqlValidatorCatalogReader catalogReader,
          RelDataTypeFactory typeFactory,
          SqlConformance conformance) {
    super(opTab, catalogReader, typeFactory, conformance);
  }

  // override SqlValidator
  @Override
  public boolean shouldExpandIdentifiers() {
    return true;
  }
}

