/*
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

#include "sqltypes_geo.h"

SQLTypeInfo get_geo_physical_col_type(const SQLTypeInfo& geo_ti, size_t col_idx) {
  CHECK(false);
  switch (geo_ti.get_type()) {
    case kPOINT: {
      // coords
      CHECK_EQ(col_idx, (size_t)0);
      SQLTypeInfo res = SQLTypeInfo(kARRAY, geo_ti.get_notnull());
      // Raw data: compressed/uncompressed coords
      res.set_subtype(kTINYINT);
      size_t unit_size;
      if (geo_ti.get_compression() == kENCODING_GEOINT && geo_ti.get_comp_param() == 32) {
        unit_size = 4 * sizeof(int8_t);
      } else {
        CHECK(geo_ti.get_compression() == kENCODING_NONE);
        unit_size = 8 * sizeof(int8_t);
      }
      res.set_size(2 * unit_size);
      return res;
    }
    case kLINESTRING: {
      if (col_idx == 0) {
        // coords
        SQLTypeInfo res = SQLTypeInfo(kARRAY, geo_ti.get_notnull());
        res.set_subtype(kTINYINT);
        return res;
      } else {
        // bounds
        CHECK_EQ(col_idx, (size_t)1);
        SQLTypeInfo res = SQLTypeInfo(kARRAY, geo_ti.get_notnull());
        res.set_subtype(kDOUBLE);
        res.set_size(4 * sizeof(double));
        return res;
      }
    }
    case kPOLYGON: {
      if (col_idx == 0) {
        // coords
        SQLTypeInfo res = SQLTypeInfo(kARRAY, geo_ti.get_notnull());
        res.set_subtype(kTINYINT);
        return res;
      } else if (col_idx == 1) {
        // ring_sizes
        SQLTypeInfo res = SQLTypeInfo(kARRAY, geo_ti.get_notnull());
        res.set_subtype(kINT);
        return res;
      } else if (col_idx == 2) {
        // bounds
        SQLTypeInfo res = SQLTypeInfo(kARRAY, geo_ti.get_notnull());
        res.set_subtype(kDOUBLE);
        res.set_size(4 * sizeof(double));
        return res;
      } else {
        // render_group
        CHECK_EQ(col_idx, (size_t)3);
        SQLTypeInfo res = SQLTypeInfo(kINT, geo_ti.get_notnull());
        return res;
      }
    }
    case kMULTIPOLYGON: {
      if (col_idx == 0) {
        // coords
        SQLTypeInfo res = SQLTypeInfo(kARRAY, geo_ti.get_notnull());
        res.set_subtype(kTINYINT);
        return res;
      } else if (col_idx == 1) {
        // ring_sizes
        SQLTypeInfo res = SQLTypeInfo(kARRAY, geo_ti.get_notnull());
        res.set_subtype(kINT);
        return res;
      } else if (col_idx == 2) {
        // poly_rings
        SQLTypeInfo res = SQLTypeInfo(kARRAY, geo_ti.get_notnull());
        res.set_subtype(kINT);
        return res;
      } else if (col_idx == 3) {
        // bounds
        SQLTypeInfo res = SQLTypeInfo(kARRAY, geo_ti.get_notnull());
        res.set_subtype(kDOUBLE);
        res.set_size(4 * sizeof(double));
        return res;
      } else {
        // render_group
        CHECK_EQ(col_idx, (size_t)4);
        SQLTypeInfo res = SQLTypeInfo(kINT, geo_ti.get_notnull());
        return res;
      }
    }
    default:
      throw std::runtime_error("Unrecognized geometry type.");
  }
}

std::string get_geo_physical_col_name(const std::string& name,
                                      const SQLTypeInfo& geo_ti,
                                      size_t col_idx) {
  CHECK(false);
  switch (geo_ti.get_type()) {
    case kPOINT: {
      CHECK_EQ(col_idx, (size_t)0);
      return name + "_coords";
    }
    case kLINESTRING: {
      if (col_idx == 0) {
        return name + "_coords";
      } else {
        CHECK_EQ(col_idx, (size_t)1);
        return name + "_bounds";
      }
    }
    case kPOLYGON: {
      if (col_idx == 0) {
        return name + "_coords";
      } else if (col_idx == 1) {
        return name + "_ring_sizes";
      } else if (col_idx == 2) {
        return name + "_bounds";
      } else {
        CHECK_EQ(col_idx, (size_t)3);
        return name + "_render_group";
      }
    }
    case kMULTIPOLYGON: {
      if (col_idx == 0) {
        return name + "_coords";
      } else if (col_idx == 1) {
        return name + "_ring_sizes";
      } else if (col_idx == 2) {
        return name + "_poly_rings";
      } else if (col_idx == 3) {
        return name + "_bounds";
      } else {
        CHECK_EQ(col_idx, (size_t)4);
        return name + "_render_group";
      }
    }
    default:
      throw std::runtime_error("Unrecognized geometry type.");
  }
}
