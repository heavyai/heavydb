/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ExecuteRenderInterface/RenderQueryUtils/thrust/ThrustGeoImplFactory.h"

#include "ExecuteRenderInterface/RenderQueryUtils/thrust/ThrustLines.h"
#include "ExecuteRenderInterface/RenderQueryUtils/thrust/ThrustPolygons.h"

namespace QueryRenderer {

template <>
std::unique_ptr<LineDataConverterInterface>
ThrustImplFactory::createGeoConverterCuda<LineDataConverterInterface>() {
  return std::make_unique<LineDataConverterImpl<ThrustDeviceSystem::kCuda>>();
}

template <>
std::unique_ptr<PolygonDataConverterInterface>
ThrustImplFactory::createGeoConverterCuda<PolygonDataConverterInterface>() {
  return std::make_unique<PolygonDataConverterImpl<ThrustDeviceSystem::kCuda>>();
}

}  // namespace QueryRenderer
