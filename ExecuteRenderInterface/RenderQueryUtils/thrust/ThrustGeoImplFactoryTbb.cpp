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
ThrustImplFactory::createGeoConverterTbb<LineDataConverterInterface>() {
  return std::make_unique<LineDataConverterImpl<ThrustDeviceSystem::kTbb>>();
}

template <>
std::unique_ptr<PolygonDataConverterInterface>
ThrustImplFactory::createGeoConverterTbb<PolygonDataConverterInterface>() {
  return std::make_unique<PolygonDataConverterImpl<ThrustDeviceSystem::kTbb>>();
}

}  // namespace QueryRenderer
