/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "HitTestBuffers.h"
#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/RenderLogger.h"
#include "QueryRenderer/Cache/ResultCache.h"
#include "QueryRenderer/GlobalRenderContext.h"
#include "QueryRenderer/QueryRendererContext.h"
#include "QueryRenderer/Rendering/QueryFramebuffer.h"
#include "QueryRenderer/Rendering/QueryIdMapPixelBuffer.h"

#include <string>

namespace QueryRenderer {

using ::gfx::DeviceContext;
using Array2df = ::gfx::Objects::Array2d<float>;

namespace {
static Array2df create_gaussian_kernel(int kernelSize, float stddev = 1.0) {
  // NOTE: it is reasonable to create the gaussian kernel as a 2D
  // array for getIdAt() since we are only using the gaussian values
  // for sums and not as coefficients. If the latter, like in performing
  // a gaussian blur in image processing, then we'd want to take advantage
  // of the fact that the gaussian kernel is seperable:
  // See:
  // https://en.wikipedia.org/wiki/Gaussian_blur
  // and
  // https://en.wikipedia.org/wiki/Separable_filter
  assert(kernelSize > 0 && kernelSize % 2 == 1);

  Array2df kernel(kernelSize, kernelSize);

  std::unique_ptr<float[]> kernel1d(new float[kernelSize]);
  int i, j;
  float stddevsq2x = 2.0 * stddev * stddev;

  // NOTE: kernel size will always be odd
  int half_kernel_size = kernelSize / 2;
  for (i = -half_kernel_size, j = 0; i <= half_kernel_size; ++i, ++j) {
    kernel1d[j] =
        std::pow(M_E, -float(i * i) / stddevsq2x) / std::sqrt(M_PI * stddevsq2x);
  }

  for (i = 0; i < kernelSize; ++i) {
    for (j = 0; j < kernelSize; ++j) {
      kernel[i][j] = kernel1d[i] * kernel1d[j];
    }
  }

  return kernel;
}
}  // namespace

HitTestBuffers::HitTestBuffers(const QueryRendererContext& ctx)
    : ctx_(ctx), pbo_gpu_(gfx::NullDeviceId), id_pixels_dirty_(false) {}
HitTestBuffers::~HitTestBuffers() {
  releasePbo();
}

HitInfo HitTestBuffers::getIdAt(uint32_t x, uint32_t y, uint32_t pixel_radius) {
  RUNTIME_EX_ASSERT(ctx_.doHitTest(),
                    "QueryRenderer " + std::string(ctx_.getRenderSessionKey()) +
                        " was not initialized for hit-testing.");

  uint64_t id{0u};
  uint32_t result_cache_id{QueryResultCache::kEmptyCacheId};

  if (id_1a_pixels_ && id_1b_pixels_ && id_2_pixels_) {
    auto const width = ctx_.getWidth();
    auto const height = ctx_.getHeight();
    if (x < width && y < height) {
      // make sure we have fully updated our cached ids first before accessing
      updateCpuCache();

      if (pixel_radius == 0) {
        id = static_cast<uint64_t>(id_1a_pixels_->get(x, y)) |
             (static_cast<uint64_t>(id_1b_pixels_->get(x, y)) << 32);
        result_cache_id = id_2_pixels_->get(x, y);
      } else {
        using KernelMap = std::unordered_map<uint32_t, Array2df>;
        static KernelMap gauss_kernels;

        auto const pixel_radius_2x_plus_1 = pixel_radius * 2 + 1;
        auto kernel_itr = gauss_kernels.find(pixel_radius);
        if (kernel_itr == gauss_kernels.end()) {
          kernel_itr = gauss_kernels
                           .emplace(pixel_radius,
                                    create_gaussian_kernel(pixel_radius_2x_plus_1, 0.75))
                           .first;
        }
        auto const& kernel = kernel_itr->second;

        using WeightMap = std::map<std::pair<uint32_t, uint64_t>, float>;
        WeightMap weight_map;

        // build the weight map
        auto const s_pixel_radius = static_cast<int32_t>(pixel_radius);
        for (uint32_t i = 0u; i < pixel_radius_2x_plus_1; ++i) {
          auto const id_y = static_cast<int32_t>(y + i) - s_pixel_radius;
          for (uint32_t j = 0u; j < pixel_radius_2x_plus_1; ++j) {
            auto const id_x = static_cast<int32_t>(x + j) - s_pixel_radius;

            // fetch IDs directly (zero if OOB)
            auto const this_id =
                static_cast<uint64_t>(id_1a_pixels_->getOrZero(id_x, id_y)) |
                (static_cast<uint64_t>(id_1b_pixels_->getOrZero(id_x, id_y)) << 32);
            auto const this_result_cache_id = id_2_pixels_->getOrZero(id_x, id_y);

            // don't include empty pixels or gaussian distro outliers
            if (this_id > 0 && kernel[i][j] > 0.0f) {
              auto weight_itr =
                  weight_map
                      .try_emplace(std::make_pair(this_result_cache_id, this_id), 0.0f)
                      .first;
              weight_itr->second += kernel[i][j];
            }
          }
        }

        // extract pixel with highest weight, if any
        auto const max_itr = std::max_element(
            weight_map.begin(), weight_map.end(), [](auto const& l, auto const& r) {
              return l.second < r.second;
            });
        if (max_itr != weight_map.end()) {
          result_cache_id = max_itr->first.first;
          id = max_itr->first.second;
        }
      }
    }
  }

  auto rtn_cache_id = static_cast<ResultCacheId>(result_cache_id >> 17);
  auto data_id = static_cast<uint8_t>((result_cache_id >> 12) & 31);
  auto node_index = static_cast<int16_t>(result_cache_id & 4095);
  return {rtn_cache_id, id, data_id, node_index};
}

void HitTestBuffers::updateFromFramebuffer(QueryFramebuffer& fbo) {
  auto const& device_context = fbo.getDeviceContext();
  // copy row/table id data over to the pbos for lazy device->host transfer
  CHECK(id_1a_pixels_ && id_2_pixels_ && pbo_1a_ && pbo_2_ &&
        device_context.getGpuId() == pbo_gpu_);

  // Initiate async transfer via PBOs
  fbo.copyRowIdBufferToPbo(pbo_1a_, true);
  if (id_1b_pixels_ && pbo_1b_) {
    fbo.copyRowIdBufferToPbo(pbo_1b_, false);
  }
  fbo.copyResultCacheIdBufferToPbo(pbo_2_);

  id_pixels_dirty_ = true;
}

/*
 * Creates pbos for a specific gpu id. The gpu id should refer to the gpu which is
 * doing the final color render pass and SMAA. If a composite is being performed, this
 * should be the compositor gpu. If a composite is unnecessary, then this is the gpuid of
 * the one-off render. If pbos exist that do not match the final_render_gpu_id argument,
 * then those pbos are released back to the pool and new ones grabbed from the pool of the
 * gpu id argument.
 */
void HitTestBuffers::createPbo(const GpuId final_render_gpu_id, int width, int height) {
  RENDER_LOG_SCOPE();
  if (pbo_gpu_ != final_render_gpu_id) {
    releasePbo();
  }

  if (!pbo_1a_ || !pbo_1b_ || !pbo_2_) {
    auto& gpu_data = ctx_.getGlobalContext().getGpuData(final_render_gpu_id);

    auto width_to_use = (width < 0 ? ctx_.getWidth() : width);
    auto height_to_use = (height < 0 ? ctx_.getHeight() : height);
    pbo_1a_wk_ = gpu_data.getInactiveIdMapPbo(width_to_use, height_to_use);
    pbo_1a_ = pbo_1a_wk_.lock();
    pbo_1b_wk_ = gpu_data.getInactiveIdMapPbo(width_to_use, height_to_use);
    pbo_1b_ = pbo_1b_wk_.lock();
    pbo_2_wk_ = gpu_data.getInactiveIdMapPbo(width_to_use, height_to_use);
    pbo_2_ = pbo_2_wk_.lock();

    pbo_gpu_ = gpu_data.getGpuId();
  }
}

void HitTestBuffers::releasePbo() {
  if (pbo_1a_ || pbo_1b_ || pbo_2_) {
    auto& gpu_data = ctx_.getGlobalContext().getGpuData(pbo_gpu_);

    if (pbo_1a_) {
      pbo_1a_ = nullptr;
      gpu_data.setIdMapPboInactive(pbo_1a_wk_);
    }

    if (pbo_1b_) {
      pbo_1b_ = nullptr;
      gpu_data.setIdMapPboInactive(pbo_1b_wk_);
    }

    if (pbo_2_) {
      pbo_2_ = nullptr;
      gpu_data.setIdMapPboInactive(pbo_2_wk_);
    }

    pbo_gpu_ = gfx::NullDeviceId;
  }
}

bool HitTestBuffers::updateCpuCache() {
  if (id_1a_pixels_ && id_2_pixels_) {
    if (id_pixels_dirty_) {
      size_t width = ctx_.getWidth();
      size_t height = ctx_.getHeight();
      CHECK(id_1a_pixels_->getWidth() == width && id_1a_pixels_->getHeight() == height &&
            id_2_pixels_->getWidth() == width && id_2_pixels_->getHeight() == height);

      if (pbo_gpu_ != gfx::NullDeviceId) {
        uint32_t* raw_ids_1a = id_1a_pixels_->getDataPtr();
        uint32_t* raw_ids_2 = id_2_pixels_->getDataPtr();
        pbo_1a_->readIdBuffer(width, height, raw_ids_1a);
        if (id_1b_pixels_ && pbo_1b_) {
          auto raw_ids_1b = id_1b_pixels_->getDataPtr();
          pbo_1b_->readIdBuffer(width, height, raw_ids_1b);
        }
        pbo_2_->readIdBuffer(width, height, raw_ids_2);
        releasePbo();
      } else {
        resetCpuCache();
      }

      id_pixels_dirty_ = false;
    }
    return true;
  }
  return false;
}

void HitTestBuffers::resize(uint32_t width, uint32_t height) {
  // resize the cpu-bound pixels that store the ids per-pixel
  if (!id_1a_pixels_) {
    id_1a_pixels_ = std::make_unique<Array2dui>(width, height);
  } else {
    id_1a_pixels_->resize(width, height);
  }

  if (!id_1b_pixels_) {
    id_1b_pixels_ = std::make_unique<Array2dui>(width, height);
  } else {
    id_1b_pixels_->resize(width, height);
  }

  if (!id_2_pixels_) {
    id_2_pixels_ = std::make_unique<Array2dui>(width, height);
  } else {
    id_2_pixels_->resize(width, height);
  }

  if (pbo_1a_) {
    CHECK(pbo_2_);
    pbo_1a_->resize(width, height);
    if (pbo_1b_) {
      pbo_1b_->resize(width, height);
    }
    pbo_2_->resize(width, height);
  }
}

void HitTestBuffers::resetCpuCache() {
  releasePbo();

  if (id_1a_pixels_) {
    id_1a_pixels_->resetToDefault();
  }
  if (id_1b_pixels_) {
    id_1b_pixels_->resetToDefault();
  }
  if (id_2_pixels_) {
    id_2_pixels_->resetToDefault();
  }
}

}  // namespace QueryRenderer
