/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <gtest/gtest.h>

#include "CudaMgr/CudaMgr.h"
#include "QueryRenderer/GlobalRenderContext.h"
#include "Tests/RenderTests/GfxDriver/GfxDriverTestFixtures.h"
#include "Tests/RenderTests/Utils/GoldenImage.h"

using namespace QueryRenderer;

namespace QueryRendererTests {

// Flag to force writing of all golden image files
static constexpr bool kRegenerateGoldenImages = false;
static constexpr gfx::RasterSampleCount kNumSamples = gfx::RasterSampleCount::k4;

template <typename GTEST_BASE,
          typename DRIVER_TYPE,
          typename SHADER_LIBRARY,
          bool USE_CUDA = false>
class QueryRendererTestBase : public GTEST_BASE {
 protected:
  void SetUp() override {
    if constexpr (USE_CUDA) {  // NOLINT
      cuda_mgr_ = std::make_unique<CudaMgr_Namespace::CudaMgr>(1);
      ASSERT_NE(nullptr, cuda_mgr_) << "Failed to create CudaMgr";
    }

    gfx_context_ =
        GfxDriverTests::GfxContextFactory<DRIVER_TYPE, SHADER_LIBRARY>::create();
    ASSERT_NE(nullptr, gfx_context_) << "Failed to create GfxContext";

    auto const device_group = cuda_mgr_
                                  ? cuda_mgr_->getDeviceGroup()
                                  : gfx_context_->getPrimaryDriver().getDeviceGroup();
    gfx_context_->createDeviceContexts(device_group);

    global_ctx_ = std::make_unique<GlobalRenderContext>(
        *gfx_context_, nullptr, cuda_mgr_.get(), 100000000, false, kNumSamples, false);
    ASSERT_NE(nullptr, global_ctx_) << "Failed to create GlobalRenderContext";

    auto const& driver = gfx_context_->getPrimaryDriver();
    ASSERT_NO_THROW(global_ctx_->init());

    golden_image_ = std::make_unique<GoldenImage>(
        std::string(RENDER_TESTS_PATH) + "QueryRenderer/golden_images/",
        driver.getType(),
        kRegenerateGoldenImages);
  }

  void TearDown() override {
    ASSERT_NO_THROW(global_ctx_ = nullptr)
        << "GlobalRenderContext destructor threw an exception";
    ASSERT_NO_THROW(gfx_context_ = nullptr) << "GfxContext destructor threw an exception";
    ASSERT_NO_THROW(cuda_mgr_ = nullptr) << "CudaMgr destructor threw an exception";
  }

  std::unique_ptr<CudaMgr_Namespace::CudaMgr> cuda_mgr_;
  std::unique_ptr<GfxContext> gfx_context_;
  std::unique_ptr<GlobalRenderContext> global_ctx_;
  std::unique_ptr<GoldenImage> golden_image_;
};

}  // namespace QueryRendererTests
