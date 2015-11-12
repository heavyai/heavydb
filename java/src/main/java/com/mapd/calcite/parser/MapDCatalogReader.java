/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.mapd.calcite.parser;

import org.apache.calcite.linq4j.tree.Expression;
import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelOptSchema;
import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.prepare.Prepare;
import org.apache.calcite.rel.RelCollation;
import org.apache.calcite.rel.RelCollations;
import org.apache.calcite.rel.RelDistribution;
import org.apache.calcite.rel.RelDistributions;
import org.apache.calcite.rel.RelFieldCollation;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.logical.LogicalTableScan;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeComparability;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rel.type.RelDataTypeFieldImpl;
import org.apache.calcite.sql.SqlAccessType;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSyntax;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.ObjectSqlType;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.validate.SqlModality;
import org.apache.calcite.sql.validate.SqlMoniker;
import org.apache.calcite.sql.validate.SqlMonikerImpl;
import org.apache.calcite.sql.validate.SqlMonikerType;
import org.apache.calcite.sql.validate.SqlMonotonicity;
import org.apache.calcite.sql.validate.SqlValidatorCatalogReader;
import org.apache.calcite.sql.validate.SqlValidatorUtil;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.Pair;
import org.apache.calcite.util.Util;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Ordering;
import com.google.common.collect.Sets;
import com.mapd.thrift.server.MapD;
import com.mapd.thrift.server.TColumnType;
import com.mapd.thrift.server.TDBInfo;
import com.mapd.thrift.server.TDatumType;
import com.mapd.thrift.server.TQueryResult;
import com.mapd.thrift.server.TTypeInfo;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.thrift.TException;
import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.protocol.TProtocol;
import org.apache.thrift.transport.TSocket;
import org.apache.thrift.transport.TTransport;
import org.apache.thrift.transport.TTransportException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * MapD implementation of {@link SqlValidatorCatalogReader} which returns tables "EMP", "DEPT",
 * "BONUS", "SALGRADE" (same as Oracle's SCOTT schema). Also two streams "ORDERS", "SHIPMENTS"; and
 * a view "EMP_20".
 */
public class MapDCatalogReader implements Prepare.CatalogReader {
  //~ Static fields/initializers ---------------------------------------------

  final static Logger logger = LoggerFactory.getLogger(MapDCatalogReader.class);

  protected static final String DEFAULT_CATALOG = "CATALOG";
  protected static String DEFAULT_SCHEMA = "mapd";

  public static final Ordering<Iterable<String>> CASE_INSENSITIVE_LIST_COMPARATOR
          = Ordering.from(String.CASE_INSENSITIVE_ORDER).lexicographical();

  //~ Instance fields --------------------------------------------------------
  protected final RelDataTypeFactory typeFactory;
  private final boolean caseSensitive;
  private final boolean elideRecord = true;
  private final Map<List<String>, MapDTable> tables;
  protected final Map<String, MapDSchema> schemas;
  private RelDataType addressType;

  //~ Constructors -----------------------------------------------------------
  /**
   * Creates a MapDCatalogReader.
   *
   * <p>
   * Caller must then call {@link #init} to populate with data.</p>
   *
   * @param typeFactory Type factory
   * @param caseSensitive boolean
   */
  public MapDCatalogReader(RelDataTypeFactory typeFactory,
          boolean caseSensitive) {
    this.typeFactory = typeFactory;
    this.caseSensitive = caseSensitive;
    if (caseSensitive) {
      tables = Maps.newHashMap();
      schemas = Maps.newHashMap();
    } else {
      tables = Maps.newTreeMap(CASE_INSENSITIVE_LIST_COMPARATOR);
      schemas = Maps.newTreeMap(String.CASE_INSENSITIVE_ORDER);
    }
  }

  /**
   * Initializes this catalog reader.
   *
   * @return MapDCatalogReader reads the catalog for this database we will need to hold catalog info
   * in the calcite server with a mechanism to overwrite, each schema will equate to one catalog
   */
  public MapDCatalogReader init() {
    return init("mapd", "HyperInteractive", "localhost", 9091, "mapd");
  }

  /**
   * Initializes this catalog reader.
   *
   * @param user
   * @param passwd
   * @param host
   * @param port
   * @param db
   * @return
   */
  public MapDCatalogReader init(String user, String passwd, String host, int port, String db) {

     DEFAULT_SCHEMA = db;

    final RelDataType intType
            = typeFactory.createSqlType(SqlTypeName.INTEGER);
    final RelDataType smallIntType
            = typeFactory.createSqlType(SqlTypeName.SMALLINT);
    final RelDataType floatType
            = typeFactory.createSqlType(SqlTypeName.FLOAT);
    final RelDataType stringType
            = typeFactory.createSqlType(SqlTypeName.VARCHAR, 50);
    final RelDataType timestampType
            = typeFactory.createSqlType(SqlTypeName.TIMESTAMP);
    final RelDataType booleanType
            = typeFactory.createSqlType(SqlTypeName.BOOLEAN);
    final RelDataType dateType
            = typeFactory.createSqlType(SqlTypeName.DATE);
    final RelDataType timeType
            = typeFactory.createSqlType(SqlTypeName.TIME);
    final RelDataType bigIntType
            = typeFactory.createSqlType(SqlTypeName.BIGINT);
    final RelDataType decimalType
            = typeFactory.createSqlType(SqlTypeName.DECIMAL);
    final RelDataType doubleType
            = typeFactory.createSqlType(SqlTypeName.DOUBLE);
    final RelDataType arrayType
            = typeFactory.createSqlType(SqlTypeName.ARRAY);

    // now with nulls
    final RelDataType intTypeNull
            = typeFactory.createTypeWithNullability(intType, true);
    final RelDataType smallIntTypeNull
            = typeFactory.createTypeWithNullability(smallIntType, true);
    final RelDataType floatTypeNull
            = typeFactory.createTypeWithNullability(floatType, true);
    final RelDataType stringTypeNull
            = typeFactory.createTypeWithNullability(stringType, true);
    final RelDataType timestampTypeNull
            = typeFactory.createTypeWithNullability(timestampType, true);
    final RelDataType booleanTypeNull
            = typeFactory.createTypeWithNullability(booleanType, true);
    final RelDataType dateTypeNull
            = typeFactory.createTypeWithNullability(dateType, true);
    final RelDataType timeTypeNull
            = typeFactory.createTypeWithNullability(timeType, true);
    final RelDataType bigIntTypeNull
            = typeFactory.createTypeWithNullability(bigIntType, true);
    final RelDataType decimalTypeNull
            = typeFactory.createTypeWithNullability(decimalType, true);
    final RelDataType doubleTypeNull
            = typeFactory.createTypeWithNullability(doubleType, true);
    final RelDataType arrayTypeNull
            = typeFactory.createTypeWithNullability(arrayType, true);

    int session = 0;
    MapD.Client client = null;
    // establish connection to mapd server

    TTransport transport;
    try {
      transport = new TSocket(host, port);

      transport.open();

      TProtocol protocol = new TBinaryProtocol(transport);

      client = new MapD.Client(protocol);

      session = client.connect(user, passwd, db);

      logger.debug("Connected session is " + session);

      MapDSchema schema = new MapDSchema(db);

      registerSchema(schema);

      logger.debug("Schema is " + db);

      // now for each db collect all tables
      List<String> ttables = client.get_tables(session);
      for (String table : ttables) {
        logger.debug("\t table  is " + table);
        MapDTable mtable = MapDTable.create(this, schema, table, false);

        // Now get tables column details
        Map<String, TColumnType> tableDescriptor = client.get_table_descriptor(session, table);

        for (Map.Entry<String, TColumnType> entry : tableDescriptor.entrySet()) {
          TColumnType value = entry.getValue();
          //logger.info(" Key String "+entry.getKey());
          logger.debug("'" + entry.getKey() + "'"
                  + " \t" + value.getCol_type().getEncoding()
                  + " \t" + value.getCol_type().getFieldValue(TTypeInfo._Fields.TYPE)
                  + " \t" + value.getCol_type().nullable
                  + " \t" + value.getCol_type().is_array
          );

          // check if it is an array
          if (value.getCol_type().is_array) {
            // set column to array
          } else {

            switch (value.getCol_type().type) {
              case SMALLINT:
                if (value.getCol_type().nullable) {
                  mtable.addColumn(entry.getKey(), smallIntTypeNull);
                } else {
                  mtable.addColumn(entry.getKey(), smallIntType);
                }
                break;
              case INT:
                if (value.getCol_type().nullable) {
                  mtable.addColumn(entry.getKey(), intTypeNull);
                } else {
                  mtable.addColumn(entry.getKey(), intType);
                }
                break;
              case BIGINT:
                if (value.getCol_type().nullable) {
                  mtable.addColumn(entry.getKey(), bigIntTypeNull);
                } else {
                  mtable.addColumn(entry.getKey(), bigIntType);
                }
                break;
              case FLOAT:
                if (value.getCol_type().nullable) {
                  mtable.addColumn(entry.getKey(), floatTypeNull);
                } else {
                  mtable.addColumn(entry.getKey(), floatType);
                }
                break;
              case DECIMAL:
                if (value.getCol_type().nullable) {
                  mtable.addColumn(entry.getKey(), decimalTypeNull);
                } else {
                  mtable.addColumn(entry.getKey(), decimalType);
                }
                break;
              case DOUBLE:
                if (value.getCol_type().nullable) {
                  mtable.addColumn(entry.getKey(), doubleTypeNull);
                } else {
                  mtable.addColumn(entry.getKey(), doubleType);
                }
                break;
              case STR:
                if (value.getCol_type().nullable) {
                  mtable.addColumn(entry.getKey(), stringTypeNull);
                } else {
                  mtable.addColumn(entry.getKey(), stringType);
                }
                break;
              case TIME:
                if (value.getCol_type().nullable) {
                  mtable.addColumn(entry.getKey(), timeTypeNull);
                } else {
                  mtable.addColumn(entry.getKey(), timeType);
                }
                break;
              case TIMESTAMP:
                if (value.getCol_type().nullable) {
                  mtable.addColumn(entry.getKey(), timestampTypeNull);
                } else {
                  mtable.addColumn(entry.getKey(), timestampType);
                }
                break;
              case DATE:
                if (value.getCol_type().nullable) {
                  mtable.addColumn(entry.getKey(), dateTypeNull);
                } else {
                  mtable.addColumn(entry.getKey(), dateType);
                }
                break;
              case BOOL:
                if (value.getCol_type().nullable) {
                  mtable.addColumn(entry.getKey(), booleanTypeNull);
                } else {
                  mtable.addColumn(entry.getKey(), booleanType);
                }
                break;
              default:
                throw new java.lang.AssertionError(value.getCol_type().type.name());
            };
          }
        }
        mtable.setRowCount(client.get_row_count(session, table));
        registerTable(mtable);
      }

      client.disconnect(session);

    } catch (TTransportException ex) {
      throw new RuntimeException("Connection failed - " + ex.toString());
    } catch (TException ex) {
      throw new RuntimeException("Connection failed - " + ex.toString());
    }

    addDefaultTestSchemas();

    return this;
  }

  /**
   * This schema adds for testing purposes the scott/tiger default schema
   */
  private MapDCatalogReader addDefaultTestSchemas() {

    final RelDataType intType
            = typeFactory.createSqlType(SqlTypeName.INTEGER);
    final RelDataType intTypeNull
            = typeFactory.createTypeWithNullability(intType, true);
    final RelDataType varchar10Type
            = typeFactory.createSqlType(SqlTypeName.VARCHAR, 10);
    final RelDataType varchar20Type
            = typeFactory.createSqlType(SqlTypeName.VARCHAR, 20);
    final RelDataType timestampType
            = typeFactory.createSqlType(SqlTypeName.TIMESTAMP);
    final RelDataType booleanType
            = typeFactory.createSqlType(SqlTypeName.BOOLEAN);
    final RelDataType rectilinearCoordType
            = typeFactory.builder().add("X", intType).add("Y", intType).build();

    // TODO jvs 12-Feb-2005: register this canonical instance with type
    // factory
    addressType
            = new ObjectSqlType(
                    SqlTypeName.STRUCTURED,
                    new SqlIdentifier("ADDRESS", SqlParserPos.ZERO),
                    false,
                    Arrays.asList(
                            new RelDataTypeFieldImpl("STREET", 0, varchar20Type),
                            new RelDataTypeFieldImpl("CITY", 1, varchar20Type),
                            new RelDataTypeFieldImpl("ZIP", 1, intType),
                            new RelDataTypeFieldImpl("STATE", 1, varchar20Type)),
                    RelDataTypeComparability.NONE);

    // Register "SALES" schema.
    MapDSchema salesSchema = new MapDSchema("SALES");
    registerSchema(salesSchema);

    // Register "EMP" table.
    final MapDTable empTable
            = MapDTable.create(this, salesSchema, "EMP", false);
    empTable.addColumn("EMPNO", intType);
    empTable.addColumn("ENAME", varchar20Type);
    empTable.addColumn("JOB", varchar10Type);
    empTable.addColumn("MGR", intTypeNull);
    empTable.addColumn("HIREDATE", timestampType);
    empTable.addColumn("SAL", intType);
    empTable.addColumn("COMM", intType);
    empTable.addColumn("DEPTNO", intType);
    empTable.addColumn("SLACKER", booleanType);
    registerTable(empTable);

    // Register "DEPT" table.
    MapDTable deptTable = MapDTable.create(this, salesSchema, "DEPT", false);
    deptTable.addColumn("DEPTNO", intType);
    deptTable.addColumn("NAME", varchar10Type);
    registerTable(deptTable);

    // Register "BONUS" table.
    MapDTable bonusTable = MapDTable.create(this, salesSchema, "BONUS", false);
    bonusTable.addColumn("ENAME", varchar20Type);
    bonusTable.addColumn("JOB", varchar10Type);
    bonusTable.addColumn("SAL", intType);
    bonusTable.addColumn("COMM", intType);
    registerTable(bonusTable);

    // Register "SALGRADE" table.
    MapDTable salgradeTable = MapDTable.create(this, salesSchema, "SALGRADE",
            false);
    salgradeTable.addColumn("GRADE", intType);
    salgradeTable.addColumn("LOSAL", intType);
    salgradeTable.addColumn("HISAL", intType);
    registerTable(salgradeTable);

    // Register "EMP_ADDRESS" table
    MapDTable contactAddressTable
            = MapDTable.create(this, salesSchema, "EMP_ADDRESS", false);
    contactAddressTable.addColumn("EMPNO", intType);
    contactAddressTable.addColumn("HOME_ADDRESS", addressType);
    contactAddressTable.addColumn("MAILING_ADDRESS", addressType);
    registerTable(contactAddressTable);

    // Register "CUSTOMER" schema.
    MapDSchema customerSchema = new MapDSchema("CUSTOMER");
    registerSchema(customerSchema);

    // Register "CONTACT" table.
    MapDTable contactTable = MapDTable.create(this, customerSchema, "CONTACT",
            false);
    contactTable.addColumn("CONTACTNO", intType);
    contactTable.addColumn("FNAME", varchar10Type);
    contactTable.addColumn("LNAME", varchar10Type);
    contactTable.addColumn("EMAIL", varchar20Type);
    contactTable.addColumn("COORD", rectilinearCoordType);
    registerTable(contactTable);

    // Register "ACCOUNT" table.
    MapDTable accountTable = MapDTable.create(this, customerSchema, "ACCOUNT",
            false);
    accountTable.addColumn("ACCTNO", intType);
    accountTable.addColumn("TYPE", varchar20Type);
    accountTable.addColumn("BALANCE", intType);
    registerTable(accountTable);

    // Register "ORDERS" stream.
    MapDTable ordersStream = MapDTable.create(this, salesSchema, "ORDERS",
            true);
    ordersStream.addColumn("ROWTIME", timestampType);
    ordersStream.addMonotonic("ROWTIME");
    ordersStream.addColumn("PRODUCTID", intType);
    ordersStream.addColumn("ORDERID", intType);
    registerTable(ordersStream);

    // Register "SHIPMENTS" stream.
    MapDTable shipmentsStream = MapDTable.create(this, salesSchema, "SHIPMENTS",
            true);
    shipmentsStream.addColumn("ROWTIME", timestampType);
    shipmentsStream.addMonotonic("ROWTIME");
    shipmentsStream.addColumn("ORDERID", intType);
    registerTable(shipmentsStream);

//    // Register "EMP_20" view.
//    // Same columns as "EMP",
//    // but "DEPTNO" not visible and set to 20 by default
//    // and "SAL" is visible but must be greater than 1000
//    MapDTable emp20View = new MapDTable(this, salesSchema.getCatalogName(),
//        salesSchema.name, "EMP_20", false) {
//      private final Table table = empTable.unwrap(Table.class);
//      private final ImmutableIntList mapping =
//          ImmutableIntList.of(0, 1, 2, 3, 4, 5, 6, 8);
//
//      @Override public RelNode toRel(ToRelContext context) {
//        // Expand to the equivalent of:
//        //   SELECT EMPNO, ENAME, JOB, MGR, HIREDATE, SAL, COMM, SLACKER
//        //   FROM EMP
//        //   WHERE DEPTNO = 20 AND SAL > 1000
//        RelNode rel = LogicalTableScan.create(context.getCluster(), empTable);
//        final RexBuilder rexBuilder = context.getCluster().getRexBuilder();
//        rel = LogicalFilter.create(rel,
//            rexBuilder.makeCall(
//                SqlStdOperatorTable.AND,
//                rexBuilder.makeCall(SqlStdOperatorTable.EQUALS,
//                    rexBuilder.makeInputRef(rel, 7),
//                    rexBuilder.makeExactLiteral(BigDecimal.valueOf(20))),
//                rexBuilder.makeCall(SqlStdOperatorTable.GREATER_THAN,
//                    rexBuilder.makeInputRef(rel, 5),
//                    rexBuilder.makeExactLiteral(BigDecimal.valueOf(1000)))));
//        final List<RelDataTypeField> fieldList =
//            rel.getRowType().getFieldList();
//        final List<Pair<RexNode, String>> projects =
//            new AbstractList<Pair<RexNode, String>>() {
//              @Override public Pair<RexNode, String> get(int index) {
//                return RexInputRef.of2(mapping.get(index), fieldList);
//              }
//
//              @Override public int size() {
//                return mapping.size();
//              }
//            };
//        return LogicalProject.create(rel, Pair.left(projects),
//            Pair.right(projects));
//      }
//
////      @Override public <T> T unwrap(Class<T> clazz) {
////        if (clazz.isAssignableFrom(ModifiableView.class)) {
////          return clazz.cast(
////              new JdbcTest.AbstractModifiableView() {
////                @Override public Table getTable() {
////                  return empTable.unwrap(Table.class);
////                }
////
////                @Override public Path getTablePath() {
////                  final ImmutableList.Builder<Pair<String, Schema>> builder =
////                      ImmutableList.builder();
////                  builder.add(Pair.<String, Schema>of(empTable.names.get(0), null));
////                  builder.add(Pair.<String, Schema>of(empTable.names.get(1), null));
////                  builder.add(Pair.<String, Schema>of(empTable.names.get(2), null));
////                  return Schemas.path(builder.build());
//////                  return empTable.names;
////                }
////
////                @Override public ImmutableIntList getColumnMapping() {
////                  return mapping;
////                }
////
////                @Override public RexNode getConstraint(RexBuilder rexBuilder,
////                    RelDataType tableRowType) {
////                  final RelDataTypeField deptnoField =
////                      tableRowType.getFieldList().get(7);
////                  final RelDataTypeField salField =
////                      tableRowType.getFieldList().get(5);
////                  final List<RexNode> nodes = Arrays.asList(
////                      rexBuilder.makeCall(SqlStdOperatorTable.EQUALS,
////                          rexBuilder.makeInputRef(deptnoField.getType(),
////                              deptnoField.getIndex()),
////                          rexBuilder.makeExactLiteral(BigDecimal.valueOf(20L),
////                              deptnoField.getType())),
////                      rexBuilder.makeCall(SqlStdOperatorTable.GREATER_THAN,
////                          rexBuilder.makeInputRef(salField.getType(),
////                              salField.getIndex()),
////                          rexBuilder.makeExactLiteral(BigDecimal.valueOf(1000L),
////                              salField.getType())));
////                  return RexUtil.composeConjunction(rexBuilder, nodes, false);
////                }
////
////                @Override public RelDataType
////                getRowType(final RelDataTypeFactory typeFactory) {
////                  return typeFactory.createStructType(
////                      new AbstractList<Map.Entry<String, RelDataType>>() {
////                        @Override public Map.Entry<String, RelDataType>
////                        get(int index) {
////                          return table.getRowType(typeFactory).getFieldList()
////                              .get(mapping.get(index));
////                        }
////
////                        @Override public int size() {
////                          return mapping.size();
////                        }
////                      }
////                  );
////                }
////              });
////        }
////        return super.unwrap(clazz);
////      }
//    };
//    salesSchema.addTable(Util.last(emp20View.getQualifiedName()));
//    emp20View.addColumn("EMPNO", intType);
//    emp20View.addColumn("ENAME", varchar20Type);
//    emp20View.addColumn("JOB", varchar10Type);
//    emp20View.addColumn("MGR", intTypeNull);
//    emp20View.addColumn("HIREDATE", timestampType);
//    emp20View.addColumn("SAL", intType);
//    emp20View.addColumn("COMM", intType);
//    emp20View.addColumn("SLACKER", booleanType);
//    registerTable(emp20View);
//
//    return this;
//  }
    return this;
  }
  //~ Methods ----------------------------------------------------------------

  /**
   *
   * @param opName
   * @param category
   * @param syntax
   * @param operatorList
   */
  @Override
  public void lookupOperatorOverloads(SqlIdentifier opName,
          SqlFunctionCategory category, SqlSyntax syntax,
          List<SqlOperator> operatorList) {
  }

  /**
   *
   * @return
   */
  @Override
  public List<SqlOperator> getOperatorList() {
    return ImmutableList.of();
  }

  /**
   *
   * @param schemaPath
   * @return
   */
  @Override
  public Prepare.CatalogReader withSchemaPath(List<String> schemaPath) {
    return this;
  }

  /**
   *
   * @param names
   * @return
   */
  @Override
  public Prepare.PreparingTable getTableForMember(List<String> names) {
    return getTable(names);
  }

  /**
   *
   * @return
   */
  @Override
  public RelDataTypeFactory getTypeFactory() {
    return typeFactory;
  }

  /**
   *
   * @param planner
   */
  @Override
  public void registerRules(RelOptPlanner planner) {
  }

  protected void registerTable(MapDTable table) {
    table.onRegister(typeFactory);
    tables.put(table.getQualifiedName(), table);
  }

  protected void registerSchema(MapDSchema schema) {
    schemas.put(schema.getSchemaName(), schema);
  }

  /**
   *
   * @param names
   * @return
   */
  @Override
  public Prepare.PreparingTable getTable(final List<String> names) {
    switch (names.size()) {
      case 1:
        // assume table in SALES schema (the original default)
        // if it's not supplied, because SqlValidatorTest is effectively
        // using SALES as its default schema.
        return tables.get(
                ImmutableList.of(DEFAULT_CATALOG, DEFAULT_SCHEMA, names.get(0)));
      case 2:
        return tables.get(
                ImmutableList.of(DEFAULT_CATALOG, names.get(0), names.get(1)));
      case 3:
        return tables.get(names);
      default:
        return null;
    }
  }

  /**
   *
   * @param typeName
   * @return
   */
  @Override
  public RelDataType getNamedType(SqlIdentifier typeName) {
    if (typeName.equalsDeep(
            addressType.getSqlIdentifier(),
            false)) {
      return addressType;
    } else {
      return null;
    }
  }

  /**
   *
   * @param names
   * @return
   */
  @Override
  public List<SqlMoniker> getAllSchemaObjectNames(List<String> names) {
    List<SqlMoniker> result;
    switch (names.size()) {
      case 0:
        // looking for catalog and schema names
        return ImmutableList.<SqlMoniker>builder()
                .add(new SqlMonikerImpl(DEFAULT_CATALOG, SqlMonikerType.CATALOG))
                .addAll(getAllSchemaObjectNames(ImmutableList.of(DEFAULT_CATALOG)))
                .build();
      case 1:
        // looking for schema names
        result = Lists.newArrayList();
        for (MapDSchema schema : schemas.values()) {
          final String catalogName = names.get(0);
          if (schema.getCatalogName().equals(catalogName)) {
            final ImmutableList<String> names1
                    = ImmutableList.of(catalogName, schema.getSchemaName());
            result.add(new SqlMonikerImpl(names1, SqlMonikerType.SCHEMA));
          }
        }
        return result;
      case 2:
        // looking for table names in the given schema
        MapDSchema schema = schemas.get(names.get(1));
        if (schema == null) {
          return Collections.emptyList();
        }
        result = Lists.newArrayList();
        for (String tableName : schema.getTableNames()) {
          result.add(
                  new SqlMonikerImpl(
                          ImmutableList.of(schema.getCatalogName(), schema.getSchemaName(),
                                  tableName),
                          SqlMonikerType.TABLE));
        }
        return result;
      default:
        return Collections.emptyList();
    }
  }

  public List<String> getSchemaName() {
    return ImmutableList.of(DEFAULT_CATALOG, DEFAULT_SCHEMA);
  }

  public RelDataTypeField field(RelDataType rowType, String alias) {
    return SqlValidatorUtil.lookupField(caseSensitive, elideRecord, rowType,
            alias);
  }

  public int fieldOrdinal(RelDataType rowType, String alias) {
    final RelDataTypeField field = field(rowType, alias);
    return field != null ? field.getIndex() : -1;
  }

  public boolean matches(String string, String name) {
    return Util.matches(caseSensitive, string, name);
  }

  public int match(List<String> strings, String name) {
    return Util.findMatch(strings, name, caseSensitive);
  }

  public RelDataType createTypeFromProjection(final RelDataType type,
          final List<String> columnNameList) {
    return SqlValidatorUtil.createTypeFromProjection(type, columnNameList,
            typeFactory, caseSensitive, elideRecord);
  }

}
// End MapDCatalogReader.java
