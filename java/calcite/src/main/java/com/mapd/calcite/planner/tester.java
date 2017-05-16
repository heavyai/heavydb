/*
 * Cool MapD License
 */
package com.mapd.calcite.planner;

import com.mapd.calcite.parser.MapDParser;
import com.mapd.calcite.parser.MapDSchema;
import com.mapd.calcite.parser.MapDSerializer;
import com.mapd.calcite.parser.MapDUser;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.schema.SchemaPlus;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.calcite.tools.FrameworkConfig;
import org.apache.calcite.tools.Frameworks;
import org.apache.calcite.tools.Planner;
import org.apache.calcite.tools.RelConversionException;
import org.apache.calcite.tools.ValidationException;
import org.slf4j.LoggerFactory;

/**
 *
 * @author michael
 */
public class tester {
    final static org.slf4j.Logger MAPDLOGGER = LoggerFactory.getLogger(tester.class);
    
    public static void main(String[] args){
        final SqlStdOperatorTable stdOpTab = SqlStdOperatorTable.instance();
//        SqlOperatorTable opTab =
//        ChainedSqlOperatorTable.of(stdOpTab,
//            new ListSqlOperatorTable(
//                ImmutableList.<SqlOperator>of(new MyCountAggFunction())));
        MapDUser mdu = new MapDUser("mapd","passwd","catalog",-1);
        MapDSchema mapd = new MapDSchema("/home/michael/mapd2/build/data", null, -1, mdu);
        final SchemaPlus rootSchema = Frameworks.createRootSchema(true);
        final FrameworkConfig config = Frameworks.newConfigBuilder()
        .defaultSchema(rootSchema.add("mapd", mapd))
        .operatorTable(stdOpTab)
        .build();

        Planner p = Frameworks.getPlanner(config);
        
        SqlNode parseR = null;
        try {
            parseR = p.parse("select * from customer where c_custkey = 1.345000 limit 5");
        } catch (SqlParseException ex) {
            Logger.getLogger(tester.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        SqlNode validateR = null;
        try {
                p.validate(parseR);
        } catch (ValidationException ex) {
            Logger.getLogger(tester.class.getName()).log(Level.SEVERE, null, ex);
        }
        RelRoot relR = null;
        try {
            relR = p.rel(validateR);
        } catch (RelConversionException ex) {
            Logger.getLogger(tester.class.getName()).log(Level.SEVERE, null, ex);
        }
        MAPDLOGGER.error ("Result was " + relR);
        MAPDLOGGER.error ("Result project() " + relR.project());
        MAPDLOGGER.error ("Result project() " + RelOptUtil.toString(relR.project()));
        MAPDLOGGER.error ("Json Version \n" + MapDSerializer.toString(relR.project()));
        
        
        // now do with MapD parser
        MapDParser mp = new MapDParser("/home/michael/mapd2/build/data", null, -1);
        mp.setUser(mdu);
        
        try {
            MAPDLOGGER.error("MapDParser result: \n"+mp.getRelAlgebra("select * from customer where c_custkey = 1.345000 limit 5", true, mdu, false));
        } catch (SqlParseException ex) {
            Logger.getLogger(tester.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ValidationException ex) {
            Logger.getLogger(tester.class.getName()).log(Level.SEVERE, null, ex);
        } catch (RelConversionException ex) {
            Logger.getLogger(tester.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
//     /** User-defined aggregate function. */
//  public static class MyCountAggFunction extends SqlAggFunction {
//    public MyCountAggFunction() {
//      super("MY_COUNT", null, SqlKind.OTHER_FUNCTION, ReturnTypes.BIGINT, null,
//          OperandTypes.ANY, SqlFunctionCategory.NUMERIC, false, false);
//    }
//
//    @SuppressWarnings("deprecation")
//    public List<RelDataType> getParameterTypes(RelDataTypeFactory typeFactory) {
//      return ImmutableList.of(typeFactory.createSqlType(SqlTypeName.ANY));
//    }
//
//    @SuppressWarnings("deprecation")
//    public RelDataType getReturnType(RelDataTypeFactory typeFactory) {
//      return typeFactory.createSqlType(SqlTypeName.BIGINT);
//    }
//
//    public RelDataType deriveType(SqlValidator validator,
//        SqlValidatorScope scope, SqlCall call) {
//      // Check for COUNT(*) function.  If it is we don't
//      // want to try and derive the "*"
//      if (call.isCountStar()) {
//        return validator.getTypeFactory().createSqlType(SqlTypeName.BIGINT);
//      }
//      return super.deriveType(validator, scope, call);
//    }
//  }
}
