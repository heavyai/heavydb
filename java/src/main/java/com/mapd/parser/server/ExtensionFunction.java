package com.mapd.parser.server;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author alex
 */
public class ExtensionFunction {

    public enum ExtArgumentType {
        Int16, Int32, Int64, Float, Double
    };

    ExtensionFunction(final List<ExtArgumentType> args, final ExtArgumentType ret) {
        this.args = args;
        this.ret = ret;
    }

    public List<ExtArgumentType> getArgs() {
        return this.args;
    }

    public ExtArgumentType getRet() {
        return this.ret;
    }

    public String toJson(final String name) {
        StringBuilder json_cons = new StringBuilder();
        json_cons.append("{");
        json_cons.append("\"name\":").append(dq(name)).append(",");
        json_cons.append("\"ret\":").append(dq(typeName(ret))).append(",");
        json_cons.append("\"args\":");
        json_cons.append("[");
        List<String> param_list = new ArrayList<String>();
        for (final ExtArgumentType arg : args) {
            param_list.add(dq(typeName(arg)));
        }
        json_cons.append(ExtensionFunctionSignatureParser.join(param_list, ","));
        json_cons.append("]");
        json_cons.append("}");
        return json_cons.toString();
    }

    private static String typeName(final ExtArgumentType type) {
        switch (type) {
            case Int16:
                return "i16";
            case Int32:
                return "i32";
            case Int64:
                return "i64";
            case Float:
                return "float";
            case Double:
                return "double";
        }
        assert false;
        return null;
    }

    private static String dq(final String str) {
        return "\"" + str + "\"";
    }

    private final List<ExtArgumentType> args;
    private final ExtArgumentType ret;
}
