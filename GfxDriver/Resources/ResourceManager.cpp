/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Resources/ResourceManager.h"

#include <algorithm>

#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/Pipeline/Pipeline.h"
#include "GfxDriver/Resources/Buffer.h"
#include "GfxDriver/Resources/HostVisibleBufferWrapper.h"
#include "GfxDriver/Resources/IndirectDrawBuffer.h"
#include "GfxDriver/Resources/LocalBufferAllocator.h"
#include "GfxDriver/Resources/RenderPass.h"
#include "GfxDriver/Resources/Texture.h"
#include "GfxDriver/Resources/VertexBuffer.h"
#include "GfxDriver/Utils/LoggingUtils.h"

#define DEBUG_LOG_RESOURCE_ADD_REMOVE 0

namespace gfx {

BufferWrapperUqPtr ResourceManager::createBuffer(
    std::string_view resource_tracking_string,
    const BufferCreateInfo& create_info,
    std::optional<BufferAllocatorShPtr> buffer_allocator_opt,
    std::optional<LoggingCallback> oom_logging_cb) {
  BufferAllocatorShPtr buffer_allocator =
      buffer_allocator_opt
          ? buffer_allocator_opt.value()
          : std::make_shared<DefaultBufferAllocator>(device_ctx_,
                                                     resource_tracking_string,
                                                     *this,
                                                     create_info,
                                                     oom_logging_cb);
  buffer_allocator->validateCreateInfo(create_info);

  BufferWrapperUqPtr buffer_wrapper;
  switch (create_info.buffer_type) {
    case BufferType::kVertexBuffer:
      buffer_wrapper = std::make_unique<VertexBuffer>(resource_tracking_string,
                                                      std::move(buffer_allocator),
                                                      create_info,
                                                      oom_logging_cb);
      break;
    case BufferType::kIndexBuffer:
      buffer_wrapper = std::make_unique<IndexBuffer>(resource_tracking_string,
                                                     std::move(buffer_allocator),
                                                     create_info,
                                                     oom_logging_cb);
      break;
    case BufferType::kIndirectDrawVertexBuffer:
      buffer_wrapper =
          std::make_unique<IndirectDrawVertexBuffer>(resource_tracking_string,
                                                     std::move(buffer_allocator),
                                                     create_info,
                                                     oom_logging_cb);
      break;
    case BufferType::kIndirectDrawIndexBuffer:
      buffer_wrapper =
          std::make_unique<IndirectDrawIndexBuffer>(resource_tracking_string,
                                                    std::move(buffer_allocator),
                                                    create_info,
                                                    oom_logging_cb);
      break;
    case BufferType::kAccelerationStructureBuffer:
    case BufferType::kShaderBindingTableBuffer:
    case BufferType::kSlabWrapperBuffer:
    case BufferType::kUnspecified:
      buffer_wrapper = std::make_unique<BufferWrapper>(resource_tracking_string,
                                                       std::move(buffer_allocator),
                                                       create_info,
                                                       oom_logging_cb);
      break;
    default:
      CHECK(false) << "Unsupported BufferType (" << create_info.buffer_type
                   << ") in createBuffer";
  }
  CHECK(buffer_wrapper);
  return buffer_wrapper;
}

HostVisibleBufferWrapperUqPtr ResourceManager::createHostVisibleBuffer(
    std::string_view resource_tracking_string,
    const HostVisibleBufferCreateInfo& create_info) {
  return convertToHostVisibleBufferImpl(
      createBuffer(resource_tracking_string, create_info));
}

HostVisibleBufferWrapperUqPtr ResourceManager::convertToHostVisibleBuffer(
    BufferWrapperUqPtr&& source_buffer) {
  // NOTE: it would probably make more sense to throw an error if the source_buffer is not
  // tagged as a kHostVisible buffer, but what if the caller wants to retain ownership of
  // the source_buffer after the throw as part of error resolution? How would we pass the
  // source_buffer rval back to the caller? One hacky way is to create a new exception
  // type that takes ownership of the source_buffer which can then be repossessed in a
  // catch.
  CHECK_EQ(source_buffer->getAccessType(), BufferAccessType::kHostVisible);
  return convertToHostVisibleBufferImpl(std::move(source_buffer));
}

void ResourceManager::destroyTexture(resource_ptr<Texture> tex) {
  removeResource(unlockResourcePtr(tex), ResourceType::kTexture);
}

void ResourceManager::destroyBaseBuffer(resource_ptr<Buffer> buffer) {
  removeResource(unlockResourcePtr(buffer), buffer->getResourceType());
}

void ResourceManager::destroyBuffer(BufferWrapperUqPtr buffer_wrapper) {
  buffer_wrapper = nullptr;
}

void ResourceManager::destroyHostVisibleBuffer(
    HostVisibleBufferWrapperUqPtr buffer_wrapper) {
  CHECK(buffer_wrapper);
  if (buffer_wrapper->source_buffer_wrapper_) {
    if (buffer_wrapper->isMapped()) {
      buffer_wrapper->unmap();
    }
    destroyBuffer(std::move(buffer_wrapper->source_buffer_wrapper_));
  }
}

void ResourceManager::destroyPipeline(resource_ptr<GraphicsPipeline> pipeline) {
  removeResource(unlockResourcePtr(pipeline), ResourceType::kPipeline);
}
void ResourceManager::destroyPipeline(resource_ptr<ComputePipeline> pipeline) {
  removeResource(unlockResourcePtr(pipeline), ResourceType::kPipeline);
}
void ResourceManager::destroyPipeline(resource_ptr<RaytracingPipeline> pipeline) {
  removeResource(unlockResourcePtr(pipeline), ResourceType::kPipeline);
}

void ResourceManager::destroyRenderPass(resource_ptr<RenderPass> render_pass) {
  removeResource(unlockResourcePtr(render_pass), ResourceType::kRenderPass);
}

void ResourceManager::destroyQueryPool(resource_ptr<QueryPool> query_pool) {
  removeResource(unlockResourcePtr(query_pool), ResourceType::kQueryPool);
}

const ResourceManager::Stats ResourceManager::getStats() const {
  Stats stats = {};
  for (auto const& resource : resources_) {
    if (resource.get()) {
      switch (resource->getResourceType()) {
        case ResourceType::kShaderProgram:
          stats.num_shader_programs++;
          break;
        case ResourceType::kShaderModule:
          stats.num_shader_modules++;
          break;
        case ResourceType::kFramebuffer:
          stats.num_framebuffers++;
          break;
        case ResourceType::kVertexArray:
          stats.num_primitive_assemblies++;
          break;
        case ResourceType::kTexture: {
          auto const* texture = static_cast<Texture*>(resource.get());
          if (texture->isArrayTexture()) {
            stats.num_texture_arrays++;
          } else {
            stats.num_textures++;
          }
          stats.bytes_used_textures += texture->getWidth() * texture->getHeight() *
                                       texture->getDepth() * texture->getNumSamples() *
                                       pixelFormatDataSize(texture->getPixelFormat());

        } break;
        case ResourceType::kBaseBuffer:
        case ResourceType::kVertexBuffer:
        case ResourceType::kIndexBuffer:
        case ResourceType::kPixelBuffer:
        case ResourceType::kIndirectDrawVertexBuffer:
        case ResourceType::kIndirectDrawIndexBuffer:
        case ResourceType::kAccelerationStructureBuffer:
        case ResourceType::kShaderBindingTableBuffer:
        case ResourceType::kSlabWrapperBuffer: {
          auto const buffer = static_cast<Buffer*>(resource.get());
          auto const buffer_type_index = static_cast<int>(buffer->getType());
          stats.num_buffers[buffer_type_index]++;
          stats.bytes_used_buffers[buffer_type_index] += buffer->getNumBytes();
        } break;
        case ResourceType::kAccelerationStructure:
          stats.num_acceleration_structures++;
          break;
        case ResourceType::kPipeline:
          stats.num_pipelines++;
          break;
        case ResourceType::kRenderPass:
          stats.num_render_passes++;
          break;
        case ResourceType::kQueryPool:
          stats.num_query_pools++;
          break;
      }
    }
  }
  return stats;
}

bool ResourceManager::hasResources() const {
  return (resources_.size() - free_rsrc_ids_.size()) > 0U;
}

Resource* ResourceManager::addResource(ResourceUqPtr&& resource) {
#if DEBUG_LOG_RESOURCE_ADD_REMOVE
  LOG(INFO) << "ResourceManager " << device_ctx_.getGpuId() << " creating "
            << to_string(resource->getResourceType()) << " resource for "
            << resource_tracking_string << " at " << std::hex << resource.get();
#endif
  Resource* raw_resource = resource.get();
  ResourceId rsrc_id;
  // do we have any resusable Resource IDs
  if (free_rsrc_ids_.size()) {
    // get a free Resource ID and check that it's in range
    rsrc_id = free_rsrc_ids_.back();
    free_rsrc_ids_.pop_back();
    CHECK(rsrc_id >= 0 && rsrc_id < resources_.size());
    // check that this slot in the list is empty, and replace
    CHECK(resources_[rsrc_id] == nullptr);
    resources_[rsrc_id] = std::move(resource);
  } else {
    // no free Resource IDs available, so allocate a new one
    rsrc_id = resources_.size();
    // add new resource to the end of the list
    resources_.emplace_back(std::move(resource));
  }
  // store the Resource ID
  resources_[rsrc_id]->setResourceId(rsrc_id);
  // done
  return raw_resource;
}

void ResourceManager::logMemorySummary(std::ostream& os) const {
  std::vector<Resource*> buffers;
  std::vector<Resource*> slab_wrapper_buffers;
  std::vector<Resource*> textures;
  size_t max_name_len = 0ULL;
  for (auto const& r : resources_) {
    if (r) {
      auto type = r->getResourceType();
      max_name_len = std::max(max_name_len, r->getTrackingData().origin.length());
      if ((type == ResourceType::kTexture) || (type == ResourceType::kPixelBuffer)) {
        textures.push_back(r.get());
      } else if (type == ResourceType::kSlabWrapperBuffer) {
        slab_wrapper_buffers.push_back(r.get());
      } else {
        buffers.push_back(r.get());
      }
    }
  }
  os << "Resource allocations on gpu " << device_ctx_.getGpuId() << "\n";

  StreamStatFormatter log_item(
      os, std::max(max_name_len, size_t(StreamStatFormatter::kDefaultNameWidth)));

  auto log_vector = [&](const std::vector<Resource*>& resources) -> uint64_t {
    uint64_t total = 0ULL;
    for (auto const& r : resources) {
      auto size = r->getGpuAllocationSize();
      if (size > 0ULL) {
        total += size;
        log_item(r->getTrackingData().origin, size);
      }
    }
    return total;
  };

  uint64_t buffer_total = 0;
  uint64_t texture_total = 0;
  uint64_t slab_wrapper_buffer_total = 0;
  if (buffers.size()) {
    log_item << "\n-- Buffers --\n";
    buffer_total = log_vector(buffers);
  }
  if (textures.size()) {
    log_item << "\n-- Textures --\n";
    texture_total = log_vector(textures);
  }
  if (slab_wrapper_buffers.size()) {
    log_item << "\n-- Slab Wrapper Buffers --\n";
    slab_wrapper_buffer_total = log_vector(slab_wrapper_buffers);
  }

  log_item << "\n-- Gpu " << device_ctx_.getGpuId() << " summary --\n";

  log_item("Buffers", buffer_total);
  log_item("Textures", texture_total);
  log_item("Total (non-slab-allocated)", buffer_total + texture_total);
  log_item("Slab Wrapper Buffers Total", slab_wrapper_buffer_total);
}

void ResourceManager::removeResource(Resource* resource, ResourceType resourceType) {
  // we must be asked to remove something
  CHECK(resource != nullptr) << "ResourceManager " << device_ctx_.getGpuId()
                             << "Asked to remove null " << to_string(resourceType)
                             << " resource";
  auto itr = std::find_if(resources_.begin(),
                          resources_.end(),
                          [this, resource, resourceType](auto& this_resource) {
                            if (this_resource.get() == resource) {
                              // and it must be of the matching type
                              CHECK(resource->getResourceType() == resourceType)
                                  << "ResourceManager " << device_ctx_.getGpuId()
                                  << "Asked to remove resource that is not a "
                                  << to_string(resourceType);
                              return true;
                            }
                            return false;
                          });
  // assert if we don't find it, or we do but it's the wrong type
  RUNTIME_EX_ASSERT(itr != resources_.end(),
                    "Failed to find given " + to_string(resourceType) + " resource");
#if DEBUG_LOG_RESOURCE_ADD_REMOVE
  LOG(INFO) << "ResourceManager " << device_ctx_.getGpuId() << " destroying "
            << to_string(resource->getResourceType()) << " resource for "
            << resource->getTrackingData().origin << " at " << std::hex << resource;
#endif
  // capture the Resource ID
  ResourceId rsrc_id = resource->getUniqueResourceId().second;
  // validate that the the ID matches where we found it
  // these should never happen unless something is badly wrong
  CHECK(rsrc_id >= 0 && rsrc_id < resources_.size())
      << "Resource ID (" << rsrc_id << ") invalid";
  CHECK(resources_[rsrc_id].get() == resource)
      << "Resource ID (" << rsrc_id << ") is not the right resource";
  // null the UqPtr which will call the resource object destructor
  // which will release any graphics API resources
  resources_[rsrc_id] = nullptr;
  // the UqPtr stays in the list as null, which marks it as reusable
  // add the relinquished Resource ID to the free list
  free_rsrc_ids_.push_back(rsrc_id);
}

void ResourceManager::cleanupResources() {
  // anything to clean up?
  if (resources_.size() == 0) {
    return;
  }

  // clean up
  uint32_t num_leaked_resources = 0;
  std::exception_ptr first_cleanup_exception;
  for (auto& resource : resources_) {
    if (resource) {
      if (++num_leaked_resources == 1) {
        LOG(WARNING) << "ResourceManager " << device_ctx_.getGpuId()
                     << " The following resources remain undestroyed at shutdown:";
      }
      LOG(WARNING) << "  " << to_string(resource->getResourceType()) << " resource at "
                   << std::hex << resource.get() << std::dec << " created for "
                   << resource->getTrackingData().origin;
      try {
        // try to clean it up, which may throw
        resource->cleanupResourceBase();
      } catch (...) {
        if (first_cleanup_exception) {
          // shit's on fire, yo!
          CHECK(false) << "Caught more than one resource clean-up exception; aborting!";
        } else {
          // catch just the first of any clean-up exceptions
          first_cleanup_exception = std::current_exception();
        }
      }
    }
  }
  // re-throw the first clean-up exception, if any
  if (first_cleanup_exception) {
    std::rethrow_exception(first_cleanup_exception);
  }
  // report any leaked resources
  if (num_leaked_resources) {
    LOG(WARNING) << "ResourceManager " << device_ctx_.getGpuId() << " had "
                 << num_leaked_resources << " at shutdown!";
  }
  resources_.clear();
}

}  // namespace gfx
