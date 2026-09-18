/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <vector>

#include "CudaMgr/CudaMgr.h"
#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/Resources/Texture.h"
#include "QueryRenderer/PerGpuData.h"

namespace QueryRenderer {

using ArrayLayerTransferredCBFunc =
    std::function<void(uint32_t,                                    // array layer index,
                       const std::vector<gfx::SemaphoreHandle>&)>;  // signal semaphores

/**
 * TextureTransferContext
 *
 * Copies 1 or more textures into one or more texture arrays.
 * Also able to copy a single texture array into a destination texture array
 *
 * Implementations use Cuda to perform the peer-to-peer copying
 *
 * Cuda resources required for the copy are always cached for the destination
 * texture arrays. Source textures can be cached as well. There is also a
 * function for copying uncached source textures
 *
 * */

class TextureTransferContext {
 public:
  // TEMP enum to identify sync requirements for copyTexturesFromDevice
  // TODO: replace with semaphores OR standardize this and use it to
  // to replace the name in the factory
  enum class SyncType { kRGBA, kID };

  // Factory function
  // Returns proper implementation based on DriverType
  static std::unique_ptr<TextureTransferContext> create(
      const GlobalRenderContext& global_ctx,
      const CudaMgr_Namespace::CudaMgr* cuda_mgr,
      const gfx::DeviceContext& device_ctx,
      std::string_view name);

  explicit TextureTransferContext(const std::string_view name) : name_{name} {}
  virtual ~TextureTransferContext() = default;

  const std::string& getName() const;

  virtual void buildResources(uint32_t width, uint32_t height) = 0;
  virtual void updateTileQueue(uint32_t width, uint32_t height) = 0;

  virtual void registerSourceDevice(
      const RootPerGpuData& gpu_data,
      const std::vector<const gfx::Texture*>& src_id_textures) = 0;

  // Copy source RGBA or ID textures from the source GPU to the destination texture arrays
  // on the compositor gpu. If source and destination GPUs are the same, simple texture
  // copies are performed
  // ID textures are cached in registerSourceDevice
  // RGBA texture is pulled from SeparateMultiSamples
  virtual void copyTexturesFromDevice(
      GpuId source_gpu_id,
      SyncType sync_type,
      const std::vector<gfx::resource_ptr<gfx::Texture>>& dst_textures) = 0;

  // Copy source texture array from source GPU to destination texture array, which will
  // be a different GPU from source
  virtual void copyTextureArrayFromDevice(
      GpuId source_gpu_id,
      const gfx::Texture& src_texture_array,
      const gfx::Texture& dst_texture,
      uint32_t num_layers_to_copy,
      ArrayLayerTransferredCBFunc layer_transfer_cb) = 0;

  virtual void waitComplete() = 0;

 protected:
  std::string name_;
};

}  // namespace QueryRenderer
