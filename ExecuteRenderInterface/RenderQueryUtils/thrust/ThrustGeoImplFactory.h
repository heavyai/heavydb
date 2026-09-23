/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ExecuteRenderInterface/RenderQueryUtils/thrust/ThrustLinesInterface.h"
#include "ExecuteRenderInterface/RenderQueryUtils/thrust/ThrustPolygonsInterface.h"
#include "QueryRenderer/Utils/thrust/ThrustDeviceSystem.h"

namespace QueryRenderer {

struct ThrustImplFactory {
 public:
  template <typename GeoConverterInterface>
  static std::unique_ptr<GeoConverterInterface> createGeoConverter(
      const ThrustDeviceSystem device_system) {
    switch (device_system) {
      case ThrustDeviceSystem::kCuda:
#ifdef HAVE_CUDA
        return createGeoConverterCuda<GeoConverterInterface>();
#else
        UNREACHABLE();
#endif
      case ThrustDeviceSystem::kTbb:
#ifndef HAVE_CUDA
        return createGeoConverterTbb<GeoConverterInterface>();
#else
        UNREACHABLE();
#endif
    }

    UNREACHABLE();
    return nullptr;
  }

 private:
#ifdef HAVE_CUDA
  template <typename GeoConverterInterface>
  static std::unique_ptr<GeoConverterInterface> createGeoConverterCuda();
#else
  template <typename GeoConverterInterface>
  static std::unique_ptr<GeoConverterInterface> createGeoConverterTbb();
#endif  // HAVE_CUDA
};

#ifdef HAVE_CUDA
template <>
std::unique_ptr<LineDataConverterInterface>
ThrustImplFactory::createGeoConverterCuda<LineDataConverterInterface>();

template <>
std::unique_ptr<PolygonDataConverterInterface>
ThrustImplFactory::createGeoConverterCuda<PolygonDataConverterInterface>();
#else
template <>
std::unique_ptr<LineDataConverterInterface>
ThrustImplFactory::createGeoConverterTbb<LineDataConverterInterface>();

template <>
std::unique_ptr<PolygonDataConverterInterface>
ThrustImplFactory::createGeoConverterTbb<PolygonDataConverterInterface>();
#endif  // HAVE_CUDA

}  // namespace QueryRenderer
