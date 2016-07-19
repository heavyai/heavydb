package com.mapd.parser.server;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 *
 * @author alex
 */
class ExtensionFunctionSignatureParser {

    static Map<String, ExtensionFunction> parse(final String file_path) throws IOException {
        File file = new File(file_path);
        FileReader fileReader = new FileReader(file);
        BufferedReader bufferedReader = new BufferedReader(fileReader);
        String line;
        Pattern r = Pattern.compile("([\\w]+) '([\\w]+) \\((.*)\\)'");
        Map<String, ExtensionFunction> sigs = new HashMap<String, ExtensionFunction>();
        while ((line = bufferedReader.readLine()) != null) {
            Matcher m = r.matcher(line);
            if (m.find()) {
                final String name = m.group(1);
                final String ret = m.group(2);
                final String cs_param_list = m.group(3);
                sigs.put(name, toSignature(ret, cs_param_list));
            }
        }
        return sigs;
    }

    static String signaturesToJson(final Map<String, ExtensionFunction> sigs) {
        List<String> json_sigs = new ArrayList<String>();
        if (sigs != null) {
            for (Map.Entry<String, ExtensionFunction> sig : sigs.entrySet()) {
                json_sigs.add(sig.getValue().toJson(sig.getKey()));
            }
        }
        return "[" + join(json_sigs, ",") + "]";
    }

    private static ExtensionFunction toSignature(final String ret, final String cs_param_list) {
        String[] params = cs_param_list.split(", ");
        List<ExtensionFunction.ExtArgumentType> args = new ArrayList<ExtensionFunction.ExtArgumentType>();
        for (final String param : params) {
            final ExtensionFunction.ExtArgumentType arg_type = deserializeType(param);
            if (arg_type != ExtensionFunction.ExtArgumentType.Void) {
                args.add(arg_type);
            }
        }
        return new ExtensionFunction(args, deserializeType(ret));
    }

    private static ExtensionFunction.ExtArgumentType deserializeType(final String type_name) {
        final String const_prefix = "const ";
        if (type_name.startsWith(const_prefix)) {
            return deserializeType(type_name.substring(const_prefix.length()));
        }
        if (type_name.equals("int32_t")) {
            return ExtensionFunction.ExtArgumentType.Int32;
        }
        if (type_name.equals("int64_t")) {
            return ExtensionFunction.ExtArgumentType.Int64;
        }
        if (type_name.equals("float")) {
            return ExtensionFunction.ExtArgumentType.Float;
        }
        if (type_name.equals("double")) {
            return ExtensionFunction.ExtArgumentType.Double;
        }
        if (type_name.equals("void")) {
            return ExtensionFunction.ExtArgumentType.Void;
        }
        assert false;
        return null;
    }

    static String join(final List<String> strs, final String sep) {
        StringBuilder sb = new StringBuilder();
        if (strs.isEmpty()) {
            return "";
        }
        sb.append(strs.get(0));
        for (int i = 1; i < strs.size(); ++i) {
            sb.append(sep).append(strs.get(i));
        }
        return sb.toString();
    }
}
