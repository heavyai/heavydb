/*
 * Copyright 2017 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.mapd.metadata;

import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeComparability;
import org.apache.calcite.rel.type.RelDataTypeFamily;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rel.type.RelDataTypePrecedenceList;
import org.apache.calcite.rel.type.StructKind;
import org.apache.calcite.sql.SqlCollation;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlIntervalQualifier;
import org.apache.calcite.sql.type.SqlTypeName;

public class GeoRelDataType implements RelDataType {

    @Override
    public boolean isStruct() {
        return false;
    }

    @Override
    public List<RelDataTypeField> getFieldList() {
        return new ArrayList<RelDataTypeField>();
    }

    @Override
    public List<String> getFieldNames() {
        return new ArrayList<String>();
    }

    @Override
    public int getFieldCount() {
        return 0;
    }

    @Override
    public StructKind getStructKind() {
        return StructKind.NONE;
    }

    @Override
    public RelDataTypeField getField(String fieldName, boolean caseSensitive, boolean elideRecord) {
        return null;
    }

    @Override
    public boolean isNullable() {
        return false;
    }

    @Override
    public RelDataType getComponentType() {
        return null;
    }

    @Override
    public RelDataType getKeyType() {
        return null;
    }

    @Override
    public RelDataType getValueType() {
        return null;
    }

    @Override
    public Charset getCharset() {
        return null;
    }

    @Override
    public SqlCollation getCollation() {
        return null;
    }

    @Override
    public SqlIntervalQualifier getIntervalQualifier() {
        return null;
    }

    @Override
    public int getPrecision() {
        return -1;
    }

    @Override
    public int getScale() {
        return -1;
    }

    @Override
    public SqlTypeName getSqlTypeName() {
        return null;
    }

    @Override
    public SqlIdentifier getSqlIdentifier() {
        return null;
    }

    @Override
    public String getFullTypeString() {
        return "Geometry";
    }

    @Override
    public RelDataTypeFamily getFamily() {
        return null;
    }

    @Override
    public RelDataTypePrecedenceList getPrecedenceList() {
        return null;
    }

    @Override
    public RelDataTypeComparability getComparability() {
        return RelDataTypeComparability.NONE;
    }

    @Override
    public boolean isDynamicStruct() {
        return false;
    }
}
