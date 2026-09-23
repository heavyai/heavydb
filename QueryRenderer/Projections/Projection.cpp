/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Projections/Projection.h"

#include <boost/algorithm/string/replace.hpp>

#include "GfxDriver/Math/Matrix2d.h"
#include "GfxDriver/Pipeline/Material.h"
#include "QueryRenderer/Projections/Utils.h"
#include "QueryRenderer/QueryRendererContext.h"

namespace QueryRenderer {

void Projection::updateShaderFunction(
    gfx::ShaderManager::Builder& builder,
    const std::string& func_name,
    const std::string& func_type,
    const ProjectionShaderShPtr& projection_shader_src) {
  // TODO(scb): add subcomponents to the Builder
  auto func_declaration =
      projection_shader_src->getFunctionDeclaration(func_name, func_type);
  auto func_body = projection_shader_src->getFunctionBody();

  auto func = func_declaration + "{\n" + func_body + "}";
  boost::replace_all(func, builder.makeTag("inTType"), func_type);

  builder.replaceFunctionDefinition(func_name, func, false);
}

bool MercatorProjection::updateFromJSONObj(const JSONLocation& json_loc) {
  const auto prev_path = this->json_path_;
  this->json_path_ = json_loc.getPathRef();
  if (!this->ctx_.isJSONCacheUpToDate(prev_path, json_loc)) {
    const auto bounds_loc = json_loc.getMember(JSONSchema_v1::Projections::kBoundsProp);
    if (bounds_loc.isValid()) {
      auto get_item_loc = [](const JSONLocation& array_loc, const size_t idx) {
        const auto item_loc = array_loc[idx];
        RUNTIME_EX_ASSERT(
            item_loc.isNumber(),
            RapidJSONUtils::createJsonParseError(item_loc, "Element must be a number."));
        return item_loc;
      };

      auto get_bounds = [get_item_loc](const JSONLocation& array_loc) {
        RUNTIME_EX_ASSERT(
            array_loc.isArray(),
            RapidJSONUtils::createJsonParseError(
                array_loc,
                "members of the \"" +
                    std::string(JSONSchema_v1::Projections::kBoundsProp) +
                    "\" property for mercator projections must contain arrays."));

        RUNTIME_EX_ASSERT(
            array_loc.size() == 2,
            RapidJSONUtils::createJsonParseError(
                array_loc,
                "Coordinate array must be of size 2 and each element must be a number."));

        return std::array<double, 2>({get_item_loc(array_loc, 0).getDouble(),
                                      get_item_loc(array_loc, 1).getDouble()});
      };

      RUNTIME_EX_ASSERT(
          bounds_loc.isObject(),
          RapidJSONUtils::createJsonParseError(
              bounds_loc,
              "The \"" + std::string(JSONSchema_v1::Projections::kBoundsProp) +
                  "\" property for mercator projections must be an object."));

      const auto x_loc = bounds_loc.getMember(JSONSchema_v1::Projections::kXProp);
      if (x_loc.isValid()) {
        x_bounds_ = get_bounds(x_loc);
      }

      const auto y_loc = bounds_loc.getMember(JSONSchema_v1::Projections::kYProp);
      if (y_loc.isValid()) {
        y_bounds_ = get_bounds(y_loc);
      }
    }
    return true;
  } else {
    return false;
  }
}

ProjectionShaderShPtrPair MercatorProjection::getShaderSrc() {
  CHECK(x_projection_shader_);
  CHECK(y_projection_shader_);
  return {x_projection_shader_, y_projection_shader_};
}

//
// MercatorProjection
//
// The Mercator projection is linear in x/longitude, with the
// normalization to screen coordinates being done purely by
// the uProjectionViewProjMatrix transform, with no conversion
// to radians required.
//
// The expression to convert latitude in degrees (WGS84) to a
// quasi-linear north/south coordinate, nominally in meters,
// is as follows:
//
//   EARTH_RADIUS_M * log(tan(45 + (latitude_degrees / 2)))
//
// We approximate the entire log(tan())) term with a polynomial
// expression, derived using the "lolremez" tool, from:
//
// https://github.com/samhocevar/lolremez
//
// Considering the equalities...
//
//   tan((pi / 2) - x) = 1 / tan(x)
//   log(1 / x) = -log(x)
//
// ...we see that...
//
//   log(tan((pi / 2) - x)) = -log(tan(x))
//
// ...then given...
//
//   x = (pi / 4) + (latitude_radians / 2)
//
// ...we see that the original expression is "odd" with respect
// to "latitude" (i.e. f(-x)=-f(x)) we can also use the technique
// described in Tutorial 3 of the lolremez docs, to optimize the
// behaviour at the range extremes, and to minimize the error term.
//
// To achieve the desired accuracy over the +/-80 degrees range,
// we are obliged to request that the tool solve for a 15th-degree
// polynomial. This gives a maximum error of 6cm in world space.
//
// The command options are therefore:
//
// --double -d 15 -r '1e-50:80^2' 'log(tan(pi/4+(sqrt(x)*pi/360)))/sqrt(x)' '1/sqrt(x)'
//
// Since the result of the log(tan()) is defined as the normalized
// linear scale of the resulting map, again no additional units
// conversion is required, and we rely on the uProjectionViewProjMatrix
// transform to handle the rest.
//

void MercatorProjection::setUniformAttributes(gfx::Material& active_material) {
  auto x_min_bounds = transformX(x_bounds_[0]);
  auto x_max_bounds = transformX(x_bounds_[1]);
  double x_scale = 1.0 / (x_max_bounds - x_min_bounds);

  auto y_min_bounds = transformY(y_bounds_[0]);
  auto y_max_bounds = transformY(y_bounds_[1]);
  double y_scale = 1.0 / (y_max_bounds - y_min_bounds);

  gfx::Math::Matrix2d<double> view_proj_matrix(
      {x_scale, 0, 0, y_scale, -1.0 * x_min_bounds, -1.0 * y_min_bounds});
  active_material.setUniformAttribute("uProjectionViewProjMatrix",
                                      view_proj_matrix.getDataArrayRef());
}

void MercatorProjection::initShaderSource() {
  x_projection_shader_ = std::make_shared<ProjectionShader>("x");
  x_projection_shader_->addUniform("dmat3x2", "uProjectionViewProjMatrix");
  x_projection_shader_->addFunctionArgument("double", "x_degrees");

  x_projection_shader_->setFunctionBody(R"glsl(
    return (x_degrees + uProjectionViewProjMatrix[2][0]) * uProjectionViewProjMatrix[0][0] * double(viewport.width);
)glsl");

  y_projection_shader_ = std::make_shared<ProjectionShader>("y");
  y_projection_shader_->addFunctionArgument("double", "y_degrees");

  y_projection_shader_->setFunctionBody(R"glsl(
    double y_degrees_squared = y_degrees * y_degrees;
    double u = 3.4610923509293134e-58;
    u = u * y_degrees_squared + -1.5769148644800835e-53;
    u = u * y_degrees_squared + 3.2614389778678079e-49;
    u = u * y_degrees_squared + -4.0419356089559425e-45;
    u = u * y_degrees_squared + 3.3400810865720711e-41;
    u = u * y_degrees_squared + -1.9384919158744696e-37;
    u = u * y_degrees_squared + 8.109756477389658e-34;
    u = u * y_degrees_squared + -2.4706635166875875e-30;
    u = u * y_degrees_squared + 5.4743860481394333e-27;
    u = u * y_degrees_squared + -8.7044049318367135e-24;
    u = u * y_degrees_squared + 9.7754101107915972e-21;
    u = u * y_degrees_squared + -6.728626361457436e-18;
    u = u * y_degrees_squared + 9.438525614268527e-15;
    u = u * y_degrees_squared + 6.6543759286927401e-11;
    u = u * y_degrees_squared + 8.8621455747926686e-7;
    u = u * y_degrees_squared + 1.7453288065056974e-2;
    double log_tan = y_degrees * u;
    return (log_tan + uProjectionViewProjMatrix[2][1]) * uProjectionViewProjMatrix[1][1] * double(viewport.height);
)glsl");
}

double MercatorProjection::transformX(const double x) {
  // Any linear value will do, and the uProjectionViewProjMatrix
  // transformation will handle the rest. Longitude, even in degrees,
  // is already linear, so just return it.
  double x_degrees = x;
  return x_degrees;

  // This is what this used to be. The conversion to radians and
  // remap to 0-2PI is superfluous
  //
  //   double in_radians = x * M_PI / 180.0;
  //   return (in_radians + M_PI) / (2.0 * M_PI);
}

double MercatorProjection::transformY(const double y) {
  // This could be just...
  //
  //   return log(tan(M_PI_4 + (y * M_PI / 360.0)));
  //
  // ...but we use the exact same approximation here, to ensure
  // that the uProjectionViewProjMatrix transformation matches.

  double y_degrees = y;
  double y_degrees_squared = y_degrees * y_degrees;
  double u = 3.4610923509293134e-58;
  u = u * y_degrees_squared + -1.5769148644800835e-53;
  u = u * y_degrees_squared + 3.2614389778678079e-49;
  u = u * y_degrees_squared + -4.0419356089559425e-45;
  u = u * y_degrees_squared + 3.3400810865720711e-41;
  u = u * y_degrees_squared + -1.9384919158744696e-37;
  u = u * y_degrees_squared + 8.109756477389658e-34;
  u = u * y_degrees_squared + -2.4706635166875875e-30;
  u = u * y_degrees_squared + 5.4743860481394333e-27;
  u = u * y_degrees_squared + -8.7044049318367135e-24;
  u = u * y_degrees_squared + 9.7754101107915972e-21;
  u = u * y_degrees_squared + -6.728626361457436e-18;
  u = u * y_degrees_squared + 9.438525614268527e-15;
  u = u * y_degrees_squared + 6.6543759286927401e-11;
  u = u * y_degrees_squared + 8.8621455747926686e-7;
  u = u * y_degrees_squared + 1.7453288065056974e-2;
  double log_tan = y_degrees * u;
  return log_tan;

  // This is what this used to be. Again, the remap to 0-2PI is
  // superfluous, and indeed bogus. Not sure what the units of the
  // log(tan()) value are, but they're certainly not radians!
  //
  //   double in_radians = y * M_PI / 180.0;
  //   double in_arg = M_PI_4 + in_radians * 0.5;
  //   return (M_PI + log(tan(in_arg))) / (2.0 * M_PI);
}

}  // namespace QueryRenderer
