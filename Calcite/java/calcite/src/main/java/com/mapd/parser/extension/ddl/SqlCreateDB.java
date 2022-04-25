package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;
import com.mapd.parser.extension.ddl.omnisci.OmniSciOptionsMap;

import org.apache.calcite.sql.SqlCreate;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.List;
import java.util.Map;

/**
 * Class that encapsulates all information associated with a CREATE DATABASE DDL command.
 */
public class SqlCreateDB extends SqlCreate implements JsonSerializableDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("CREATE_DB", SqlKind.OTHER_DDL);

  @Expose
  private String command;
  @Expose
  private String name;
  @Expose
  private boolean ifNotExists;
  @Expose
  OmniSciOptionsMap options;

  public SqlCreateDB(final SqlParserPos pos,
          final boolean ifNotExists,
          final String name,
          OmniSciOptionsMap optionsMap) {
    super(OPERATOR, pos, false, ifNotExists);
    this.command = OPERATOR.getName();
    this.name = name;
    this.ifNotExists = ifNotExists;
    this.options = optionsMap;
  }

  @Override
  public List<SqlNode> getOperandList() {
    return null;
  }

  @Override
  public String toString() {
    return toJsonString();
  }
}
