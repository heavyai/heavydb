/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ExecuteRenderInterface/RenderQueryUtils/thrust/ThrustGeoImplFactory.h"
#include "ExecuteRenderInterface/RenderQueryUtils/thrust/ThrustLinesInterface.h"
#include "ExecuteRenderInterface/RenderQueryUtils/thrust/ThrustPolygonsInterface.h"

namespace QueryRenderer {

template <typename GeoConverterInterface>
class ThrustGeoConverter {
 public:
  ThrustGeoConverter(DataMgrThrustContext&& thrust_context)
      : thrust_context_(std::move(thrust_context)) {
#ifdef HAVE_CUDA
    impl_ = ThrustImplFactory::createGeoConverter<GeoConverterInterface>(
        ThrustDeviceSystem::kCuda);
#else
    impl_ = ThrustImplFactory::createGeoConverter<GeoConverterInterface>(
        ThrustDeviceSystem::kTbb);
#endif  // HAVE_CUDA
  }

  //
  // Stage 1
  //
  // Abstraction around stage 1 of thrust-based render result processing
  // for converting query results in a query output buffer to a render
  // buffer formatted rendering. Stage 1 generically consists of various
  // statistics gathering (strides/offsets/etc) and prefix sums to properly
  // allocate/resize render buffers in a packed format.
  //

  template <typename... Targs>
  void ConvertStage1(Targs&&... args) {
    impl_->ConvertStage1(thrust_context_, std::forward<Targs>(args)...);
  }

  //
  // Stage 2
  //
  // This completes the buffer processing by populating the render buffers
  // with the data gathered from the query output buffer(s) and returns
  // the data structures required for the draw process.
  //
  template <typename... Targs>
  void ConvertStage2(Targs&&... args) {
    impl_->ConvertStage2(thrust_context_, std::forward<Targs>(args)...);
  }

 private:
  DataMgrThrustContext thrust_context_;
  std::unique_ptr<GeoConverterInterface> impl_;
};

using PolygonDataConverter = ThrustGeoConverter<PolygonDataConverterInterface>;
using LineDataConverter = ThrustGeoConverter<LineDataConverterInterface>;

}  // namespace QueryRenderer
