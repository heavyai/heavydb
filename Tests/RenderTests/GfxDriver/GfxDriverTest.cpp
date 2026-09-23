/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Tests/RenderTests/GfxDriver/GfxDriverTestFixtures.h"

#include <memory>
#include <set>

#include "GfxDriver/Drivers/Vulkan/Commands/StagingContext.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanResourceManager.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanTexture.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/VulkanMemoryMgr.h"
#include "GfxDriver/Drivers/Vulkan/VulkanMemoryUtils.h"

#include "GfxDriver/Commands/CommandList.h"
#include "GfxDriver/GfxContext.h"
#include "GfxDriver/Objects/TileBuilder.h"
#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/Resources/AttachmentManager.h"
#include "GfxDriver/Resources/HostVisibleBufferWrapper.h"
#include "GfxDriver/Resources/SinkedPtr.h"
#include "Logger/Logger.h"
#include "Shared/scope.h"
#include "Tests/LogCaptureTestHelper.h"
#include "Tests/RenderTests/Utils/AttachmentUtils.h"
#include "Tests/RenderTests/Utils/ShaderUtils.h"
#include "Tests/TestHelpers.h"

namespace GfxDriverTests {

static std::string_view kTestFlushName{"GfxDriverTests"};

/*
 * GfxContext tests
 *
 * Create a GfxContext and ensure that it initializes properly. These are non-fixture
 * based tests
 */
TEST(GfxContextTest, CanInitVulkan) {
  auto library = std::make_unique<Library>();
  ASSERT_NE(nullptr, library) << "Failed to create Library";
  EmptyShaderLibrary::init(*library);

  // temporary env-var to control ray-tracing initialization
  // here, the default must be true
  // please forgive any and all double negatives
  auto* heavyai_allow_raytracing_init = getenv("HEAVYAI_ALLOW_RAYTRACING_INIT");
  if (heavyai_allow_raytracing_init) {
    std::cout << "**** DEBUG **** GfxContextTest: HEAVYAI_ALLOW_RAYTRACING_INIT = "
              << heavyai_allow_raytracing_init << std::endl;
  }
  const bool allow_raytracing_init = heavyai_allow_raytracing_init
                                         ? (std::stoi(heavyai_allow_raytracing_init) == 1)
                                         : true;

  std::unique_ptr<GfxContext> gfx_context;
  EXPECT_NO_THROW(gfx_context = std::make_unique<GfxContext>(DriverType::kVulkan,
                                                             kGfxUsage,
                                                             std::move(library),
                                                             60000u,
                                                             nullptr,
                                                             allow_raytracing_init));
  auto const& driver = gfx_context->getPrimaryDriver();
  EXPECT_EQ(std::string("Vulkan"), driver.getName());
  EXPECT_NE(0u, driver.getNumGpus());
  EXPECT_NO_THROW(gfx_context = nullptr);
}

/*
 * DeviceContext creation tests
 *
 * Uses the GfxDriver test fixture to exercise the DeviceContext creation APIs
 * using fresh driver instances for each test. These are typed tests so all driver
 * types will be tested.
 */
TYPED_TEST(TypedDriverTest, CanCreateDeviceContexts) {
  auto uuids = this->driver_->getUUIDs();
  ASSERT_NE(0UL, uuids.size())
      << "No UUIDs return from getUUIDs(), no valid Vulkan devices?";

  auto device_contexts = this->createDeviceContexts(uuids, this->mock_cuda_ids_);

  // Check size
  ASSERT_EQ(device_contexts.size(), uuids.size());

  // loop and destroy (easier for debugging than just clearing the vector or letting it
  // go out of scope)
  for (auto& device_context : device_contexts) {
    this->driver_->destroyDeviceContext(std::move(device_context));
    device_context = nullptr;
  }
}

TYPED_TEST(TypedDriverTest, CanRecreateDriverAndContexts) {
  auto uuids = this->driver_->getUUIDs();
  ASSERT_NE(0UL, uuids.size())
      << "No UUIDs return from getUUIDs(), no valid Vulkan devices?";

  auto device_contexts = this->createDeviceContexts(uuids, this->mock_cuda_ids_);
  ASSERT_EQ(device_contexts.size(), uuids.size());
  this->destroyDeviceContexts(device_contexts);

  // destroy and recreate the driver instance
  this->TearDown();
  this->SetUp();

  device_contexts = this->createDeviceContexts(uuids, this->mock_cuda_ids_);
  ASSERT_EQ(device_contexts.size(), uuids.size());
  this->destroyDeviceContexts(device_contexts);
}

// DeviceContext creation failure test (bad UUID)
TYPED_TEST(TypedDriverTest, DeviceContextCreationBadUUIDTest) {
  EXPECT_ANY_THROW((void)this->driver_->createDeviceContext(heavyai::empty_uuid, 0));
}

/*
 * DeviceContext API tests
 *
 * Uses the DeviceContext test fixture to exercise the DeviceContext API using fresh
 * DeviceContexts for each registered test.
 * These are typed tests so all driver types will be tested.
 */
TYPED_TEST(TypedDeviceContextTest, GetDeviceProperties) {
  std::set<DeviceId> device_ids;
  for (size_t i = 0; i < this->device_contexts_.size(); ++i) {
    EXPECT_EQ(this->device_contexts_[i]->getGpuUUID(), this->uuids_[i]);
    EXPECT_TRUE(this->device_contexts_[i]->getLimits().areValid());
    // Vulkan driver relies on cuda id being set externally

    if constexpr (std::is_same_v<TypeParam, VulkanDriver>) {  // NOLINT
      EXPECT_EQ(this->device_contexts_[i]->getGpuId(), this->mock_cuda_ids_[i]);
    }

    EXPECT_TRUE(device_ids.insert(this->device_contexts_[i]->getGpuId()).second);
  }
}

/*
 * Vulkan DeviceContext specific tests
 */
TEST_F(VulkanDeviceContextTest, GetVulkanDeviceContextProperties) {
  for (auto& dc : this->device_contexts_) {
    auto* vulkan_dc = dynamic_cast<VulkanDeviceContext*>(dc.get());
    ASSERT_NE(nullptr, vulkan_dc);
    // TODO: figure out how test against VK_NULL_HANDLE
    EXPECT_NE(nullptr, vulkan_dc->getHandle());
    EXPECT_NE(nullptr, vulkan_dc->getPhysicalDeviceHandle());
  }
}

using VulkanMemoryCreateWithUsageBitTest =
    VulkanDeviceContextTestWithParam<VkBufferUsageFlagBits>;

TEST_P(VulkanMemoryCreateWithUsageBitTest, WithUsageBit) {
  constexpr uint64_t mem_size = 1024 * 1024;
  auto const& buffer_usage = GetParam();

  for (auto& dc : this->device_contexts_) {
    auto* vulkan_dc = dynamic_cast<VulkanDeviceContext*>(dc.get());
    ASSERT_NE(vulkan_dc, nullptr);

    auto& memory_mgr = vulkan_dc->getMemoryManager();
    bool device_supports_export =
        any_bits_set(dc->getCapabilityBits() & DeviceCapabilityBits::kBufferMemoryExport);

    // create info
    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = mem_size;
    buffer_info.usage = buffer_usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkExternalMemoryBufferCreateInfo external_memory_buffer_ci = {};
    if (device_supports_export) {
      external_memory_buffer_ci.sType =
          VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
      external_memory_buffer_ci.handleTypes =
          VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
      buffer_info.pNext = &external_memory_buffer_ci;
    }

    // create it
    VkBuffer resource_handle = {};
    VkResult result =
        vkCreateBuffer(vulkan_dc->getHandle(), &buffer_info, nullptr, &resource_handle);
    ASSERT_EQ(result, VK_SUCCESS);

    // memory requirements
    VkMemoryRequirements vk_memory_requirements = {};
    vkGetBufferMemoryRequirements(
        vulkan_dc->getHandle(), resource_handle, &vk_memory_requirements);

    // allocate memory
    VulkanMemoryMgr::allocation_ptr vulkan_allocation;
    EXPECT_NO_THROW(vulkan_allocation =
                        memory_mgr.alloc("Alloc test buffer",
                                         vk_memory_requirements,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                         device_supports_export));

    ASSERT_NE(nullptr, vulkan_allocation);
    auto allocation_type = vulkan_allocation->getAllocationType();
    if (device_supports_export) {
      EXPECT_EQ(allocation_type, VulkanAllocation::AllocationType::kExportable);
    } else {
      EXPECT_EQ(allocation_type, VulkanAllocation::AllocationType::kGeneric);
    }
    EXPECT_EQ(mem_size, vulkan_allocation->size());

    // Bind the buffer to the memory allocation to ensure the block is valid
    result = vkBindBufferMemory(
        vulkan_dc->getHandle(), resource_handle, vulkan_allocation->getHandle(), 0);
    EXPECT_EQ(result, VK_SUCCESS);

    // Destroy the buffer and free the memory
    vkDestroyBuffer(vulkan_dc->getHandle(), resource_handle, nullptr);

    EXPECT_NO_THROW(memory_mgr.free(std::move(vulkan_allocation)));
    EXPECT_EQ(nullptr, vulkan_allocation);
  }
}

// Instantiate the tests
INSTANTIATE_TEST_SUITE_P(AllocateVulkanMemory,
                         VulkanMemoryCreateWithUsageBitTest,
                         ::testing::Values(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                           VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                           VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                           VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                           VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                                           VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT,
                                           VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT));

TEST_F(VulkanDeviceContextTest, OutOfDeviceMemoryTest) {
  auto* vulkan_dc = dynamic_cast<VulkanDeviceContext*>(this->device_contexts_[0].get());
  ASSERT_NE(vulkan_dc, nullptr);
  auto& memory_mgr = vulkan_dc->getMemoryManager();
  auto memory_budget = vulkan_dc->getMemoryBudget();

  // Suppress validation messages as a memory allocation failure triggers them
  VulkanPlatform::pushSuppressValidationMessages();

  // Try to create a buffer that *should* OOM without exceeding total heap size
  VkBufferCreateInfo buffer_info = {};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = memory_budget.available + memory_budget.used;
  buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkBuffer resource_handle = {};
  auto result =
      vkCreateBuffer(vulkan_dc->getHandle(), &buffer_info, nullptr, &resource_handle);
  ASSERT_EQ(result, VK_SUCCESS);

  // memory requirements
  VkMemoryRequirements vk_memory_requirements = {};
  vkGetBufferMemoryRequirements(
      vulkan_dc->getHandle(), resource_handle, &vk_memory_requirements);

  auto oom_logger = [](std::ostream& os) { os << "Test OOM logging callback\n"; };
  VulkanMemoryMgr::allocation_ptr vulkan_allocation;
  EXPECT_THROW(vulkan_allocation = memory_mgr.alloc("OOM Test buffer",
                                                    vk_memory_requirements,
                                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                                    false,
                                                    false,
                                                    -1,
                                                    oom_logger),
               OutOfGpuMemoryError);
  {
    auto memory_hog = VulkanGpuMemoryHog(*vulkan_dc, 16000000, std::cout);

    // Now try to create a big texture
    resource_ptr<Texture> texture;
    EXPECT_THROW(texture = vulkan_dc->getResourceManager().createTexture(
                     "OOM test texture",
                     2048,
                     2048,
                     1,
                     PixelFormat::kRGBA8,
                     1,
                     false,
                     ImageUsageBits::kColorAttachmentBit,
                     get_default_sampler_state_for_format(PixelFormat::kRGBA8)),
                 OutOfGpuMemoryError);

    if (texture) {
      vulkan_dc->getResourceManager().destroyTexture(std::move(texture));
    }
  }
  VulkanPlatform::popSuppressValidationMessages();

  EXPECT_EQ(vulkan_allocation, nullptr);
  vkDestroyBuffer(vulkan_dc->getHandle(), resource_handle, nullptr);
}

/*
 * ResourceManager tests
 */

// Test creating a vertex buffer. This is also covered by the BufferWrapper test below
// but can be expanded to cover any VBO specific API
TYPED_TEST(TypedDeviceContextTest, CanCreateVertexBuffer) {
  constexpr uint64_t buffer_size = 1024;
  for (auto& device : this->device_contexts_) {
    auto& resource_mgr = device->getResourceManager();
    BufferWrapperUqPtr vertex_buffer;
    std::string tracking_string("TestCanCreateVertexBuffer");
    EXPECT_NO_THROW(vertex_buffer =
                        resource_mgr.createBuffer(tracking_string,
                                                  {BufferType::kVertexBuffer,
                                                   buffer_size,
                                                   BufferUsageBits::kNone,
                                                   BufferAccessType::kDeviceLocal}));
    ASSERT_NE(nullptr, vertex_buffer);
    EXPECT_EQ(vertex_buffer->getTrackingData().origin, tracking_string);
    EXPECT_NO_THROW(resource_mgr.destroyBuffer(std::move(vertex_buffer)));
    EXPECT_EQ(nullptr, vertex_buffer);
  }
}

TYPED_TEST(TypedDeviceContextTest, BufferUpdateSubDataTest) {
  std::vector<uint32_t> host_data_1 = {0, 1, 2, 3, 4};
  std::vector<uint32_t> host_data_2 = {5, 6, 7, 8};
  std::vector<uint32_t> compare_data = {0, 1, 2, 3, 4, 5, 6, 7, 8};

  auto data_size_1 = host_data_1.size() * sizeof(uint32_t);
  auto data_size_2 = host_data_2.size() * sizeof(uint32_t);
  for (auto& device : this->device_contexts_) {
    auto& resource_mgr = device->getResourceManager();
    resource_ptr<Buffer> data_buffer;
    ScopeGuard assert_cleanup = [&]() {
      if (data_buffer) {
        resource_mgr.destroyBaseBuffer(std::move(data_buffer));
      }
    };
    ASSERT_NO_THROW(data_buffer = resource_mgr.createBaseBuffer(
                        "Data", {BufferType::kUnspecified, data_size_1 + data_size_2}));
    ASSERT_NE(nullptr, data_buffer);

    ASSERT_NO_THROW(data_buffer->updateSubData(host_data_1.data(), data_size_1, 0));
    ASSERT_NO_THROW(
        data_buffer->updateSubData(host_data_2.data(), data_size_2, data_size_1));

    std::vector<uint32_t> host_combined(host_data_1.size() + host_data_2.size(), 0);
    EXPECT_NO_THROW(
        data_buffer->getData(host_combined.data(), data_size_1 + data_size_2));
    EXPECT_EQ(compare_data, host_combined);
  }
}

TYPED_TEST(TypedDeviceContextTest, CopyAndFillBufferTest) {
  constexpr uint64_t kNumValues = 1024;
  constexpr uint64_t kBufferSize = kNumValues * sizeof(uint32_t);
  for (auto& device : this->device_contexts_) {
    std::vector<uint32_t> host_src_values(kNumValues, 0u);
    std::vector<uint32_t> host_dst_values(kNumValues, 0u);
    std::iota(host_src_values.begin(), host_src_values.end(), 0u);

    auto& resource_mgr = device->getResourceManager();
    resource_ptr<Buffer> src_buffer, dst_buffer;
    ScopeGuard assert_cleanup = [&]() {
      if (src_buffer) {
        resource_mgr.destroyBaseBuffer(std::move(src_buffer));
      }
      if (dst_buffer) {
        resource_mgr.destroyBaseBuffer(std::move(dst_buffer));
      }
    };
    EXPECT_NO_THROW(src_buffer = resource_mgr.createBaseBuffer(
                        "Test", {BufferType::kUnspecified, kBufferSize}));
    ASSERT_NE(nullptr, src_buffer);
    EXPECT_NO_THROW(dst_buffer = resource_mgr.createBaseBuffer(
                        "Test", {BufferType::kUnspecified, kBufferSize}));
    ASSERT_NE(nullptr, dst_buffer);

    // set values in source buffer
    EXPECT_NO_THROW(src_buffer->updateSubData(host_src_values.data(), kBufferSize, 0));

    // read value back to make sure buffer was set properly
    EXPECT_NO_THROW(src_buffer->getData(host_dst_values.data(), kBufferSize));
    ASSERT_EQ(host_src_values, host_dst_values);

    // copy values to dst buffer
    auto& cmd_list = device->getCommandList();
    cmd_list.copyBuffer(*src_buffer, *dst_buffer, kBufferSize)
        .flush(kTestFlushName, CommandList::SubmitType::kWaitComplete);

    // ensure copy worked
    std::fill(host_dst_values.begin(), host_dst_values.end(), 0u);
    EXPECT_NO_THROW(dst_buffer->getData(host_dst_values.data(), kBufferSize));
    ASSERT_EQ(host_src_values, host_dst_values);

    // fill source buffer with 1s, copy to dst, read and compare
    cmd_list.fillBuffer(*src_buffer, 1u, kBufferSize)
        .flush(kTestFlushName, CommandList::SubmitType::kWaitComplete);
    // TODO: memory barrier would allow chaining the commands in a single submit
    cmd_list.copyBuffer(*src_buffer, *dst_buffer, kBufferSize)
        .flush(kTestFlushName, CommandList::SubmitType::kWaitComplete);

    std::fill(host_src_values.begin(), host_src_values.end(), 1u);
    std::fill(host_dst_values.begin(), host_dst_values.end(), 0u);
    EXPECT_NO_THROW(dst_buffer->getData(host_dst_values.data(), kBufferSize));
    ASSERT_EQ(host_src_values, host_dst_values);

    EXPECT_NO_THROW(resource_mgr.destroyBaseBuffer(std::move(src_buffer)));
    ASSERT_EQ(src_buffer, nullptr);
    EXPECT_NO_THROW(resource_mgr.destroyBaseBuffer(std::move(dst_buffer)));
    ASSERT_EQ(dst_buffer, nullptr);
  }
}

// Tests creating host-visible buffers.
TYPED_TEST(TypedDeviceContextTest, CanCreateHostVisibleBuffers) {
  static constexpr std::array<BufferType, 3> generic_buffer_types = {
      BufferType::kVertexBuffer,
      BufferType::kIndirectDrawVertexBuffer,
      BufferType::kIndirectDrawIndexBuffer};
  constexpr uint64_t buffer_size = 1024;
  for (auto& device : this->device_contexts_) {
    auto& resource_mgr = device->getResourceManager();

    for (auto buffer_type : generic_buffer_types) {
      BufferWrapperUqPtr buffer_wrapper;
      HostVisibleBufferWrapperUqPtr host_visible_buffer_wrapper;
      ScopeGuard assert_cleanup = [&]() {
        if (buffer_wrapper) {
          resource_mgr.destroyBuffer(std::move(buffer_wrapper));
        }
        if (host_visible_buffer_wrapper) {
          resource_mgr.destroyHostVisibleBuffer(std::move(host_visible_buffer_wrapper));
        }
      };

      // Test host-mapping

      // Tests the convert API
      std::string tracking_string("HostVisibleBufferConvertTest_" +
                                  to_string(buffer_type));
      EXPECT_NO_THROW(buffer_wrapper =
                          resource_mgr.createBuffer(tracking_string,
                                                    {buffer_type,
                                                     buffer_size,
                                                     BufferUsageBits::kNone,
                                                     BufferAccessType::kHostVisible}));
      ASSERT_NE(nullptr, buffer_wrapper);
      EXPECT_EQ(buffer_wrapper->getNumBytes(), buffer_size);
      EXPECT_EQ(buffer_wrapper->getAccessType(), BufferAccessType::kHostVisible);
      EXPECT_EQ(buffer_wrapper->isMappable(), true);

      EXPECT_NO_THROW(
          host_visible_buffer_wrapper =
              resource_mgr.convertToHostVisibleBuffer(std::move(buffer_wrapper)));
      ASSERT_EQ(nullptr, buffer_wrapper);

      EXPECT_EQ(host_visible_buffer_wrapper->isMapped(), false);
      void* mapped_ptr{nullptr};
      host_visible_buffer_wrapper->map(&mapped_ptr);
      EXPECT_NE(nullptr, mapped_ptr);
      EXPECT_EQ(host_visible_buffer_wrapper->isMapped(), true);
      host_visible_buffer_wrapper->unmap();
      EXPECT_EQ(host_visible_buffer_wrapper->isMapped(), false);

      buffer_wrapper = host_visible_buffer_wrapper->releaseSourceBuffer();
      EXPECT_NE(nullptr, buffer_wrapper);

      EXPECT_NO_THROW(resource_mgr.destroyBuffer(std::move(buffer_wrapper)));
      ASSERT_EQ(nullptr, buffer_wrapper);

      EXPECT_NO_THROW(
          resource_mgr.destroyHostVisibleBuffer(std::move(host_visible_buffer_wrapper)));
      ASSERT_EQ(nullptr, host_visible_buffer_wrapper);

      // Now tests the direct create/destroy API
      EXPECT_NO_THROW(host_visible_buffer_wrapper = resource_mgr.createHostVisibleBuffer(
                          tracking_string, {buffer_type, buffer_size}));
      ASSERT_NE(nullptr, host_visible_buffer_wrapper);
      EXPECT_EQ(host_visible_buffer_wrapper->getSourceBufferWrapper().getNumBytes(),
                buffer_size);
      EXPECT_EQ(host_visible_buffer_wrapper->getSourceBufferWrapper().getAccessType(),
                BufferAccessType::kHostVisible);
      EXPECT_EQ(host_visible_buffer_wrapper->getSourceBufferWrapper().isMappable(), true);

      EXPECT_EQ(host_visible_buffer_wrapper->isMapped(), false);
      mapped_ptr = nullptr;
      host_visible_buffer_wrapper->map(&mapped_ptr);
      EXPECT_NE(nullptr, mapped_ptr);
      EXPECT_EQ(host_visible_buffer_wrapper->isMapped(), true);
      host_visible_buffer_wrapper->unmap();
      EXPECT_EQ(host_visible_buffer_wrapper->isMapped(), false);

      EXPECT_NO_THROW(
          resource_mgr.destroyHostVisibleBuffer(std::move(host_visible_buffer_wrapper)));
      ASSERT_EQ(nullptr, host_visible_buffer_wrapper);
    }
  }
}

// Test resource_ptr, shared_resource_ptr, with generic Buffer types
TYPED_TEST(TypedDeviceContextTest, ResourcePtrBufferWrapperTest) {
  static constexpr std::array<BufferType, 3> generic_buffer_types = {
      BufferType::kVertexBuffer,
      BufferType::kIndirectDrawVertexBuffer,
      BufferType::kIndirectDrawIndexBuffer};
  constexpr uint64_t buffer_size = 1024;
  for (auto& device : this->device_contexts_) {
    auto& resource_mgr = device->getResourceManager();
    auto initial_resource_stats = resource_mgr.getStats();

    for (auto buffer_type : generic_buffer_types) {
      BufferWrapperUqPtr buffer_wrapper;
      std::string tracking_string("ResourcePtrBufferWrapperTest_" +
                                  to_string(buffer_type));
      // Test unique resource_ptr
      EXPECT_NO_THROW(buffer_wrapper = resource_mgr.createBuffer(
                          tracking_string, {buffer_type, buffer_size}));
      ASSERT_NE(nullptr, buffer_wrapper);
      EXPECT_EQ(buffer_wrapper->getTrackingData().origin, tracking_string);
      EXPECT_NO_THROW(resource_mgr.destroyBuffer(std::move(buffer_wrapper)));
      EXPECT_EQ(nullptr, buffer_wrapper);
    }
    EXPECT_EQ(initial_resource_stats, resource_mgr.getStats());
  }
}

TYPED_TEST(TypedDeviceContextTest, GetBufferDeviceAddressTest) {
  static constexpr uint64_t kBufferSize = 64;
  bool did_test_run = false;
  for (auto& device : this->device_contexts_) {
    if (any_bits_set(device->getCapabilityBits() &
                     DeviceCapabilityBits::kBufferDeviceAddress)) {
      auto& resource_mgr = device->getResourceManager();
      BufferWrapperUqPtr buffer;
      std::string tracking_string("GetBufferDeviceAddressTest");
      ASSERT_NO_THROW(buffer =
                          resource_mgr.createBuffer(tracking_string,
                                                    {BufferType::kVertexBuffer,
                                                     kBufferSize,
                                                     BufferUsageBits::kDeviceAddressBit,
                                                     BufferAccessType::kDeviceLocal}));
      ASSERT_NE(nullptr, buffer);
      EXPECT_EQ(buffer->getTrackingData().origin, tracking_string);
      auto address = buffer->getBuffer().getDeviceAddress();
      EXPECT_GT(address, 0ul);
      EXPECT_NO_THROW(resource_mgr.destroyBuffer(std::move(buffer)));
      EXPECT_EQ(nullptr, buffer);
    }
    did_test_run = true;
  }
  if (!did_test_run) {
    // flag test as skipped if no devices support the capability
    GTEST_SKIP() << "No devices support buffer device address capability";
  }
}

ImageUsageBits pixel_format_to_image_usage_bits(PixelFormat format) {
  switch (format) {
    case PixelFormat::kR8:
    case PixelFormat::kRG8:
    case PixelFormat::kRGBA8:
    case PixelFormat::kBGRA8:
    case PixelFormat::kR32UI:
    case PixelFormat::kR32I:
      return ImageUsageBits::kColorAttachmentBit;
    case PixelFormat::kDepth:
    case PixelFormat::kDepthHighP:
    case PixelFormat::kDepthStencil:
    case PixelFormat::kDepthStencilHighP:
      return ImageUsageBits::kDepthStencilAttachmentBit;
    case PixelFormat::kCOUNT:
      CHECK(false);
  }
  UNREACHABLE();
  return ImageUsageBits::kNone;
}

// Test create / destroy of textures and texture arrays with all required formats. Should
// be expanded in the future for more API coverage (SamplerState etc)
TYPED_TEST(TypedDeviceContextTest, CanCreateTextureTest) {
  static constexpr std::array<PixelFormat, static_cast<size_t>(PixelFormat::kCOUNT)>
      pixel_formats{{PixelFormat::kR8,
                     PixelFormat::kRG8,
                     PixelFormat::kRGBA8,
                     PixelFormat::kBGRA8,
                     PixelFormat::kR32UI,
                     PixelFormat::kR32I,
                     PixelFormat::kDepth,
                     PixelFormat::kDepthHighP,
                     PixelFormat::kDepthStencil,
                     PixelFormat::kDepthStencilHighP}};

  static std::set<PixelFormat> exportable_format_set{PixelFormat::kRGBA8,
                                                     PixelFormat::kR32UI};

  constexpr uint32_t width = 512;
  constexpr uint32_t height = 386;  // test non-power of 2
  for (auto& device : this->device_contexts_) {
    bool device_supports_export = false;
    if constexpr (std::is_same_v<TypeParam, VulkanDriver>) {  // NOLINT
      device_supports_export = any_bits_set(device->getCapabilityBits() &
                                            DeviceCapabilityBits::kImageMemoryExport);
      if (!device_supports_export) {
        std::cerr << "Image memory export not supported on device. Skipping export test."
                  << std::endl;
      }
    }

    auto& resource_mgr = device->getResourceManager();
    auto initial_resource_stats = resource_mgr.getStats();
    for (auto const& pixel_format : pixel_formats) {
      auto usage_bits = pixel_format_to_image_usage_bits(pixel_format);
      bool should_export = false;

      if constexpr (std::is_same_v<TypeParam, VulkanDriver>) {  // NOLINT
        should_export =
            device_supports_export && exportable_format_set.count(pixel_format);
      }

      if (should_export) {
        usage_bits = usage_bits | ImageUsageBits::kExternalApiBit;
      }

      // texture (single layer)
      {
        resource_ptr<Texture> texture;
        std::string tracking_string("CanCreateTextureTest_" + to_string(pixel_format));
        EXPECT_NO_THROW(texture = resource_mgr.createTexture(
                            tracking_string,
                            width,
                            height,
                            1,
                            pixel_format,
                            1,
                            false,
                            usage_bits,
                            get_default_sampler_state_for_format(pixel_format)));
        ASSERT_NE(nullptr, texture);
        EXPECT_EQ(texture->getTrackingData().origin, tracking_string);
        EXPECT_EQ(texture->getPixelFormat(), pixel_format);
        EXPECT_EQ(texture->getWidth(), width);
        EXPECT_EQ(texture->getHeight(), height);

        if (should_export) {
          auto* vk_texture = static_cast<VulkanTexture*>(texture.get());
          auto* vk_alloc = vk_texture->getMemoryAllocation();
          ASSERT_EQ(vk_alloc->getAllocationType(),
                    VulkanAllocation::AllocationType::kExportable);
          auto* vk_exp_alloc = static_cast<VulkanExportableAllocation*>(vk_alloc);
          auto fd = vk_exp_alloc->exportHandle();
          EXPECT_NE(fd, 0);
          if (fd) {
            close(fd);
          }
        }

        EXPECT_NO_THROW(resource_mgr.destroyTexture(std::move(texture)));
        EXPECT_EQ(nullptr, texture);
      }

      // texture array
      {
        constexpr uint32_t array_depth = 4;
        resource_ptr<Texture> texture_array;
        std::string tracking_string("CanCreateTextureArrayTest_" +
                                    to_string(pixel_format));
        EXPECT_NO_THROW(texture_array = resource_mgr.createTexture(
                            tracking_string,
                            width,
                            height,
                            array_depth,
                            pixel_format,
                            1,
                            true,
                            usage_bits,
                            get_default_sampler_state_for_format(pixel_format)));
        ASSERT_NE(nullptr, texture_array);
        EXPECT_EQ(texture_array->getTrackingData().origin, tracking_string);
        EXPECT_EQ(texture_array->getPixelFormat(), pixel_format);
        EXPECT_EQ(texture_array->getWidth(), width);
        EXPECT_EQ(texture_array->getHeight(), height);
        EXPECT_EQ(texture_array->getDepth(), array_depth);

        if (should_export) {
          auto* vk_texture_array = static_cast<VulkanTexture*>(texture_array.get());
          auto* vk_alloc = vk_texture_array->getMemoryAllocation();
          ASSERT_EQ(vk_alloc->getAllocationType(),
                    VulkanAllocation::AllocationType::kExportable);
          auto* vk_exp_alloc = static_cast<VulkanExportableAllocation*>(vk_alloc);
          auto fd = vk_exp_alloc->exportHandle();
          EXPECT_NE(fd, 0);
          if (fd) {
            close(fd);
          }
        }

        EXPECT_NO_THROW(resource_mgr.destroyTexture(std::move(texture_array)));
        EXPECT_EQ(nullptr, texture_array);
      }
    }
    EXPECT_EQ(initial_resource_stats, resource_mgr.getStats());
  }
}

namespace {

std::vector<uint8_t> make_ramp_pixels(const uint32_t width,
                                      const uint32_t height,
                                      const uint32_t layer_count) {
  std::vector<uint8_t> pixels(width * height * layer_count * 4);
  auto* pixel_data = pixels.data();
  for (uint32_t l = 0; l < layer_count; l++) {
    for (uint32_t y = 0; y < height; y++) {
      for (uint32_t x = 0; x < width; x++) {
        *pixel_data++ = static_cast<uint8_t>(255.0f * float(x) / float(width));
        *pixel_data++ = static_cast<uint8_t>(255.0f * float(y) / float(height));
        *pixel_data++ = static_cast<uint8_t>(255.0f * float(l) / float(layer_count));
        *pixel_data++ = (uint8_t)255;
      }
    }
  }
  return pixels;
}

std::vector<uint8_t> make_black_pixels(const uint32_t width,
                                       const uint32_t height,
                                       const uint32_t layer_count) {
  std::vector<uint8_t> pixels(width * height * layer_count * 4);
  std::memset(pixels.data(), 0, pixels.size());
  return pixels;
}

static constexpr gfx::ClearTextureValue kColor(1.0f, 0.5f, 0.25f, 0.75f);
static constexpr std::array<uint8_t, 4> kColorBytes{255, 127, 64, 191};

std::vector<uint8_t> make_color_pixels(const uint32_t width,
                                       const uint32_t height,
                                       const uint32_t layer_count) {
  std::vector<uint8_t> pixels(width * height * layer_count * 4);
  auto* pixel_data = pixels.data();
  for (uint32_t l = 0; l < layer_count; l++) {
    for (uint32_t y = 0; y < height; y++) {
      for (uint32_t x = 0; x < width; x++) {
        *pixel_data++ = kColorBytes[0];
        *pixel_data++ = kColorBytes[1];
        *pixel_data++ = kColorBytes[2];
        *pixel_data++ = kColorBytes[3];
      }
    }
  }
  return pixels;
}

}  // namespace

TYPED_TEST(TypedDeviceContextTest, FramebufferAttachmentClear) {
  static constexpr uint32_t kWidth = 512;
  static constexpr uint32_t kHeight = 512;
  static constexpr PixelFormat kColorFormat = PixelFormat::kRGBA8;

  for (auto& device : this->device_contexts_) {
    auto& resource_mgr = device->getResourceManager();

    // Create AttachmentManager and Textures
    // attachment manager and textures
    auto attachments =
        build_attachments(resource_mgr,
                          kWidth,
                          kHeight,
                          {{kColorFormat, Framebuffer::Attachment::kColor0},
                           {kColorFormat, Framebuffer::Attachment::kColor1}});

    resource_ptr<RenderPass> render_pass;
    resource_ptr<Framebuffer> framebuffer;
    ScopeGuard assert_cleanup = [&]() {
      if (framebuffer) {
        resource_mgr.destroyFramebuffer(std::move(framebuffer));
      }
      if (render_pass) {
        resource_mgr.destroyRenderPass(std::move(render_pass));
      }
      for (auto& texture : attachments.textures) {
        if (texture) {
          resource_mgr.destroyTexture(std::move(texture));
        }
      }
    };

    // Create a RenderPass
    // We'te going to prefill the attachments with a ramp, so don't clear them
    ASSERT_NO_THROW(render_pass = resource_mgr.createRenderPass(
                        "FramebufferAttachmentClear test",
                        attachments.attachment_mgr.getLayout(),
                        RenderPass::ClearBits::kNone,
                        ImageLayout::kShaderReadOnly,
                        ImageLayout::kTransferSrc,
                        {}));
    ASSERT_NO_THROW(framebuffer =
                        resource_mgr.createFramebuffer("FramebufferAttachmentClear test",
                                                       *render_pass,
                                                       attachments.attachment_mgr,
                                                       kWidth,
                                                       kHeight,
                                                       1));

    auto ramp_pixels = make_ramp_pixels(kWidth, kHeight, 1);
    auto black_pixels = make_black_pixels(kWidth, kHeight, 1);
    std::vector<uint8_t> read_pixels(kWidth * kHeight *
                                     pixelFormatDataSize(kColorFormat));

    // Fill attachment textures
    for (auto& texture : attachments.textures) {
      ASSERT_NO_THROW(
          texture->setPixels(kWidth, kHeight, 1, kColorFormat, ramp_pixels.data()));
    }

    // Clear attachment kColor0
    // clearFramebufferAttachment must occur within a RenderPass
    device->getCommandExecutor().setDefaultViewportAndRenderArea(0, 0, kWidth, kHeight);
    ASSERT_NO_THROW(
        device->getCommandList()
            .beginRenderPass(*render_pass, *framebuffer)
            .clearFramebufferAttachment(*framebuffer, Framebuffer::Attachment::kColor1)
            .endRenderPass()
            .flush(kTestFlushName));

    // read the pixels back and ensure color0 is still ramp and color1 is black
    ASSERT_NO_THROW(attachments.textures[0]->getPixels(
        kWidth, kHeight, 1, kColorFormat, read_pixels.data(), read_pixels.size()));
    EXPECT_TRUE(read_pixels == ramp_pixels);

    ASSERT_NO_THROW(attachments.textures[1]->getPixels(
        kWidth, kHeight, 1, kColorFormat, read_pixels.data(), read_pixels.size()));
    EXPECT_TRUE(read_pixels == black_pixels);

    // destroy everything
    EXPECT_NO_THROW(resource_mgr.destroyFramebuffer(std::move(framebuffer)));
    EXPECT_NO_THROW(resource_mgr.destroyRenderPass(std::move(render_pass)));
    for (auto& texture : attachments.textures) {
      EXPECT_NO_THROW(resource_mgr.destroyTexture(std::move(texture)));
    }
  }
}

TYPED_TEST(TypedDeviceContextTest, TextureSetGetAndClear) {
  static constexpr uint32_t kFullWidth = 512;
  static constexpr uint32_t kFullHeight = 512;
  static constexpr PixelFormat kPixelFormat = PixelFormat::kRGBA8;
  for (auto& device : this->device_contexts_) {
    auto& resource_mgr = device->getResourceManager();

    resource_ptr<Texture> texture;
    ASSERT_NO_THROW(texture = resource_mgr.createTexture("Test TextureSetGetAndClear",
                                                         kFullWidth,
                                                         kFullHeight,
                                                         1,
                                                         kPixelFormat,
                                                         1,
                                                         false,
                                                         ImageUsageBits::kNone,
                                                         TextureSamplerState(),
                                                         nullptr));

    ScopeGuard assert_cleanup = [&]() {
      if (texture) {
        resource_mgr.destroyTexture(std::move(texture));
      }
    };

    auto ramp_pixels = make_ramp_pixels(kFullWidth, kFullHeight, 1);
    auto black_pixels = make_black_pixels(kFullWidth, kFullHeight, 1);
    auto color_pixels = make_color_pixels(kFullWidth, kFullHeight, 1);

    // Test setPixels and getPixels
    ASSERT_NO_THROW(
        texture->setPixels(kFullWidth, kFullHeight, 1, kPixelFormat, ramp_pixels.data()));

    std::vector<uint8_t> read_pixels(kFullWidth * kFullHeight *
                                     pixelFormatDataSize(kPixelFormat));
    ASSERT_NO_THROW(texture->getPixels(kFullWidth,
                                       kFullHeight,
                                       1,
                                       kPixelFormat,
                                       read_pixels.data(),
                                       read_pixels.size()));

    // Must pass or clear test is meaningless
    ASSERT_TRUE(read_pixels == ramp_pixels);

    // Test clearPixels and getPixels
    ASSERT_NO_THROW(texture->clearPixels());
    ASSERT_NO_THROW(texture->getPixels(kFullWidth,
                                       kFullHeight,
                                       1,
                                       kPixelFormat,
                                       read_pixels.data(),
                                       read_pixels.size()));
    EXPECT_TRUE(read_pixels == black_pixels);

    // Test clearPixelsToValue and getPixels
    ASSERT_NO_THROW(texture->clearPixelsToValue(kColor));
    ASSERT_NO_THROW(texture->getPixels(kFullWidth,
                                       kFullHeight,
                                       1,
                                       kPixelFormat,
                                       read_pixels.data(),
                                       read_pixels.size()));
    EXPECT_TRUE(read_pixels == color_pixels);

    EXPECT_NO_THROW(resource_mgr.destroyTexture(std::move(texture)));
  }
}

TYPED_TEST(TypedDeviceContextTest, TextureArraySetGetAndClear) {
  static constexpr uint32_t kFullWidth = 512;
  static constexpr uint32_t kFullHeight = 512;
  static constexpr uint32_t kLayerCount = 4;
  static constexpr PixelFormat kPixelFormat = PixelFormat::kRGBA8;
  for (auto& device : this->device_contexts_) {
    auto& resource_mgr = device->getResourceManager();

    resource_ptr<Texture> texture_array;
    ASSERT_NO_THROW(texture_array = resource_mgr.createTexture(
                        "Test TextureArraySetGetAndClear",
                        kFullWidth,
                        kFullHeight,
                        kLayerCount,
                        kPixelFormat,
                        1,
                        true,
                        ImageUsageBits::kNone,
                        get_default_sampler_state_for_format(kPixelFormat)));

    ScopeGuard assert_cleanup = [&]() {
      if (texture_array) {
        resource_mgr.destroyTexture(std::move(texture_array));
      }
    };

    // Just make the cpu buffers extra wide to accommodate the extra layers
    auto ramp_pixels = make_ramp_pixels(kFullWidth, kFullHeight, kLayerCount);
    auto black_pixels = make_black_pixels(kFullWidth, kFullHeight, kLayerCount);

    // Test setPixels and getPixels
    ASSERT_NO_THROW(texture_array->setPixels(
        kFullWidth, kFullHeight, kLayerCount, kPixelFormat, ramp_pixels.data()));

    std::vector<uint8_t> read_pixels(kFullWidth * kFullHeight * kLayerCount *
                                     pixelFormatDataSize(kPixelFormat));
    ASSERT_NO_THROW(texture_array->getPixels(kFullWidth,
                                             kFullHeight,
                                             kLayerCount,
                                             kPixelFormat,
                                             read_pixels.data(),
                                             read_pixels.size()));

    // Must pass or clear test is meaningless
    ASSERT_TRUE(read_pixels == ramp_pixels);

    // Test clearPixels and getPixels
    ASSERT_NO_THROW(texture_array->clearPixels());
    ASSERT_NO_THROW(texture_array->getPixels(kFullWidth,
                                             kFullHeight,
                                             kLayerCount,
                                             kPixelFormat,
                                             read_pixels.data(),
                                             read_pixels.size()));

    EXPECT_TRUE(read_pixels == black_pixels);

    EXPECT_NO_THROW(resource_mgr.destroyTexture(std::move(texture_array)));
  }
}

/*
 * ImageViews
 */
TEST_F(VulkanDeviceContextTest, CanCreateImageViews) {
  constexpr uint32_t kWidth = 128;
  constexpr uint32_t kHeight = 128;
  constexpr PixelFormat kTexturePixelFormat = PixelFormat::kRGBA8;
  constexpr PixelFormat kViewPixelFormat = PixelFormat::kR32UI;

  for (auto& device : this->device_contexts_) {
    auto& resource_mgr = device->getResourceManager();
    resource_ptr<Texture> texture;

    ASSERT_NO_THROW(texture = resource_mgr.createTexture("Test CanCreateImageViews",
                                                         kWidth,
                                                         kHeight,
                                                         1,
                                                         kTexturePixelFormat,
                                                         1,
                                                         false,
                                                         ImageUsageBits::kMutableViewBit,
                                                         TextureSamplerState(),
                                                         nullptr));

    ScopeGuard assert_cleanup = [&]() {
      if (texture) {
        resource_mgr.destroyTexture(std::move(texture));
      }
    };

    ASSERT_TRUE(texture->hasView(0));

    Texture::ViewCreateResult result = {0u, false};
    constexpr uint32_t kViewId = 1u;
    ASSERT_NO_THROW(result = texture->createView(kViewId, kViewPixelFormat));
    EXPECT_NE(result.first, 0u);
    EXPECT_EQ(result.second, false);

    EXPECT_TRUE(texture->hasView(kViewId));
    ResourceHandle view_handle = 0u;
    ASSERT_NO_THROW(view_handle = texture->getViewHandle(kViewId));
    ASSERT_NE(view_handle, 0u);
    PixelFormat view_format{};
    ASSERT_NO_THROW(view_format = texture->getViewPixelFormat(kViewId));
    EXPECT_EQ(view_format, kViewPixelFormat);
    EXPECT_EQ(texture->getPixelFormat(), kTexturePixelFormat);

    bool did_destroy_view = false;
    EXPECT_NO_THROW(did_destroy_view = texture->destroyView(kViewId));
    EXPECT_EQ(did_destroy_view, true);
    EXPECT_FALSE(texture->hasView(kViewId));
    EXPECT_THROW(texture->getViewHandle(kViewId), RenderError);
    EXPECT_THROW(texture->getViewPixelFormat(kViewId), RenderError);

    EXPECT_NO_THROW(resource_mgr.destroyTexture(std::move(texture)));
  }
}

/*
 * Framebuffer
 */
TYPED_TEST(TypedDeviceContextTest, CanCreateAndResizeFramebuffer) {
  constexpr uint32_t size_1 = 256;
  constexpr uint32_t size_2 = 300;
  for (auto& device : this->device_contexts_) {
    auto& resource_mgr = device->getResourceManager();
    auto initial_resource_stats = resource_mgr.getStats();
    AttachmentManager attachment_mgr;

    resource_ptr<Texture> texture;
    resource_ptr<RenderPass> render_pass;
    resource_ptr<Framebuffer> framebuffer;
    ScopeGuard assert_cleanup = [&]() {
      if (render_pass) {
        resource_mgr.destroyRenderPass(std::move(render_pass));
      }
      if (framebuffer) {
        resource_mgr.destroyFramebuffer(std::move(framebuffer));
      }
      if (texture) {
        resource_mgr.destroyTexture(std::move(texture));
      }
    };

    // Attachment texture
    ASSERT_NO_THROW(texture = resource_mgr.createTexture(
                        "Test Framebuffer Attachment",
                        size_1,
                        size_1,
                        1,
                        PixelFormat::kRGBA8,
                        1,
                        false,
                        ImageUsageBits::kColorAttachmentBit,
                        get_default_sampler_state_for_format(PixelFormat::kRGBA8)));
    attachment_mgr.setAttachment(Framebuffer::Attachment::kColor0, texture.get());

    // RenderPass
    ASSERT_NO_THROW(render_pass =
                        resource_mgr.createRenderPass("Test RenderPass",
                                                      attachment_mgr.getLayout(),
                                                      gfx::RenderPass::ClearBits::kAll,
                                                      ImageLayout::kUndefined,
                                                      ImageLayout::kAttachment));

    // Framebuffer
    ASSERT_NO_THROW(
        framebuffer = resource_mgr.createFramebuffer(
            "Test Framebuffer", *render_pass, attachment_mgr, size_1, size_1, 1));
    EXPECT_EQ(resource_mgr.getStats().num_framebuffers,
              initial_resource_stats.num_framebuffers + 1);

    EXPECT_NO_THROW(framebuffer->resize(size_2, size_2));

    EXPECT_NO_THROW(resource_mgr.destroyRenderPass(std::move(render_pass)));
    EXPECT_NO_THROW(resource_mgr.destroyFramebuffer(std::move(framebuffer)));
    attachment_mgr.clear();
    EXPECT_NO_THROW(resource_mgr.destroyTexture(std::move(texture)));
    EXPECT_EQ(resource_mgr.getStats(), initial_resource_stats);
  }
}

/*
 * Parameterized tests
 */
/*
 * Vulkan Shaders and Materials
 */

using VulkanShaderTest =
    DeviceContextTest<::testing::Test, VulkanDriver, GfxDriverTestShaderLibrary>;
TEST_F(VulkanShaderTest, CanCreateShaderWithBuildersVulkan) {
  ShaderCacheShPtrVector caches;
  ASSERT_NO_THROW(
      caches = this->gfx_context_->getShaderManager().createCacheVectorFromTemplate(
          {{"DriverTests/canCreateShaderTest.vert"},
           {"DriverTests/canCreateShaderTest.frag"}}));
  ASSERT_TRUE(validate_shader_caches(caches));

  for (auto& device : this->device_contexts_) {
    EXPECT_TRUE(device->getLimits().areValid());
    auto& vk_resource_mgr =
        static_cast<VulkanResourceManager&>(device->getResourceManager());
    auto initial_resource_stats = vk_resource_mgr.getStats();
    resource_ptr<VulkanShaderModule> shader_module_vert, shader_module_frag;
    ScopeGuard assert_cleanup = [&]() {
      if (shader_module_vert) {
        vk_resource_mgr.destroyShaderModule(std::move(shader_module_vert));
      }
      if (shader_module_frag) {
        vk_resource_mgr.destroyShaderModule(std::move(shader_module_frag));
      }
    };

    EXPECT_NO_THROW(shader_module_vert = vk_resource_mgr.createShaderModule(
                        "Test Vert ShaderModule", caches[0]));
    EXPECT_NO_THROW(shader_module_frag = vk_resource_mgr.createShaderModule(
                        "Test Frag ShaderModule", caches[1]));
    ASSERT_NE(shader_module_vert, nullptr);
    ASSERT_NE(shader_module_frag, nullptr);
    EXPECT_EQ(vk_resource_mgr.getStats().num_shader_modules,
              initial_resource_stats.num_shader_modules + 2);

    EXPECT_NO_THROW(vk_resource_mgr.destroyShaderModule(std::move(shader_module_vert)));
    EXPECT_NO_THROW(vk_resource_mgr.destroyShaderModule(std::move(shader_module_frag)));
    EXPECT_EQ(shader_module_vert, nullptr);
    EXPECT_EQ(shader_module_frag, nullptr);
    EXPECT_EQ(vk_resource_mgr.getStats(), initial_resource_stats);
  }
}

using VulkanMaterialTest =
    DeviceContextTest<::testing::Test, VulkanDriver, GfxDriverTestShaderLibrary>;
TEST_F(VulkanMaterialTest, CanUpdateDescriptors) {
  ShaderCacheShPtrVector caches;
  ASSERT_NO_THROW(
      caches = this->gfx_context_->getShaderManager().createCacheVectorFromTemplate(
          {{"DriverTests/fullScreenTriangle.vert"},
           {"DriverTests/descriptorTest.frag"}}));
  ASSERT_TRUE(validate_shader_caches(caches));

  for (auto& device : this->device_contexts_) {
    auto& vk_resource_mgr =
        static_cast<VulkanResourceManager&>(device->getResourceManager());

    static constexpr uint32_t kTextureWidth = 64;
    static constexpr uint32_t kTextureHeight = 64;
    static constexpr uint32_t kTextureArrayDepth = 4;
    static constexpr int kVectorSize = 4;
    static constexpr PixelFormat kPixelFormat = PixelFormat::kRGBA8;

    resource_ptr<Texture> texture;
    resource_ptr<Texture> texture_array;
    std::vector<resource_ptr<Texture>> array_of_texture_arrays(kVectorSize);
    std::unique_ptr<Material> material;

    ScopeGuard resource_cleanup = [&]() {
      material = nullptr;
      if (texture) {
        vk_resource_mgr.destroyTexture(std::move(texture));
      }
      if (texture_array) {
        vk_resource_mgr.destroyTexture(std::move(texture_array));
      }
      for (auto& texture_array : array_of_texture_arrays) {
        if (texture_array) {
          vk_resource_mgr.destroyTexture(std::move(texture_array));
        }
      }
    };

    ASSERT_NO_THROW(texture = vk_resource_mgr.createTexture(
                        "Test CanUpdateDescriptors",
                        kTextureWidth,
                        kTextureHeight,
                        1,
                        kPixelFormat,
                        1,
                        false,
                        ImageUsageBits::kNone,
                        get_default_sampler_state_for_format(kPixelFormat),
                        nullptr));

    auto create_texture_array = [&]() -> resource_ptr<Texture> {
      return vk_resource_mgr.createTexture(
          "Test CanUpdateDescriptors",
          kTextureWidth,
          kTextureHeight,
          kTextureArrayDepth,
          kPixelFormat,
          1,
          true,
          ImageUsageBits::kNone,
          get_default_sampler_state_for_format(kPixelFormat),
          nullptr);
    };

    ASSERT_NO_THROW(texture_array = create_texture_array());

    for (auto& texture_array : array_of_texture_arrays) {
      ASSERT_NO_THROW(texture_array = create_texture_array());
    }

    // material
    material = vk_resource_mgr.createMaterial("CanUpdateDescriptors Test", caches);
    ASSERT_NE(material, nullptr);

    material->setSamplerAttribute("sampler2d", *texture);
    material->setSamplerAttribute("sampler2dArray", *texture_array);
    material->setSamplerArrayAttribute("arrayOfSampler2dArray", array_of_texture_arrays);

    material->updateDescriptorSets();
  }
}

/*
 * Vulkan Texture Pixel Manipulation
 */

namespace {

void set_pixels(StagingContext& staging_context,
                Texture* texture,
                const std::vector<uint8_t>& pixels,
                const uint64_t full_image_size_bytes,
                const uint32_t layer_count,
                const uint32_t set_width,
                const uint32_t set_height,
                const uint32_t full_width,
                const uint32_t full_height) {
  CHECK(texture);
  auto locked_staging_buffer =
      staging_context.acquireStagingBuffer(full_image_size_bytes);
  auto* buffer_data = locked_staging_buffer.buffer;
  CHECK(buffer_data);
  ScopeGuard release_staging = [&] {
    staging_context.releaseStagingBuffer(std::move(locked_staging_buffer), texture);
  };
  auto const src_line_bytes = set_width * 4;
  auto const dst_line_bytes = full_width * 4;
  auto const src_layer_bytes = set_width * set_height * 4;
  auto const dst_layer_bytes = full_width * full_height * 4;
  for (uint32_t l = 0; l < layer_count; l++) {
    auto const* src_line = pixels.data() + (src_layer_bytes * l);
    auto* dst_line = static_cast<unsigned char*>(buffer_data) + (dst_layer_bytes * l);
    for (uint32_t y = 0; y < set_height; y++) {
      std::memcpy(dst_line, src_line, src_line_bytes);
      src_line += src_line_bytes;
      dst_line += dst_line_bytes;
    }
  }
}

}  // namespace

//
// Test texture transfer using StagingContext directly
//

TEST_F(VulkanDeviceContextTest, StagingContextTest) {
  static constexpr uint32_t kFullWidth = 1024;
  static constexpr uint32_t kFullHeight = 1024;
  static constexpr uint32_t kRegionWidth = 1023;
  static constexpr uint32_t kRegionHeight = 1023;
  for (auto& device : this->device_contexts_) {
    EXPECT_TRUE(device->getLimits().areValid());
    auto& vk_resource_mgr =
        static_cast<VulkanResourceManager&>(device->getResourceManager());
    auto const& vk_device = static_cast<const VulkanDeviceContext&>(*device);
    auto& staging = vk_device.getStagingContext();

    // create some content
    auto ramp_pixels = make_ramp_pixels(kFullWidth, kFullHeight, 1);
    auto region_ramp_pixels = make_ramp_pixels(kRegionWidth, kRegionHeight, 1);

    // image sizes
    const uint64_t full_image_size_bytes = kFullWidth * kFullHeight * 4;
    const uint64_t region_image_size_bytes = kRegionWidth * kRegionHeight * 4;

    // prepare to get pixels
    std::vector<uint8_t> full_get_pixels(full_image_size_bytes);
    std::vector<uint8_t> region_get_pixels(region_image_size_bytes);

    // create an empty texture (usage: sampled + transfer source + transfer dest)
    resource_ptr<Texture> texture;
    ASSERT_NO_THROW(texture = vk_resource_mgr.createTexture("simple texture",
                                                            kFullWidth,
                                                            kFullHeight,
                                                            1,
                                                            PixelFormat::kRGBA8,
                                                            1,
                                                            false,
                                                            ImageUsageBits::kNone,
                                                            TextureSamplerState(),
                                                            nullptr));
    auto& vk_texture = static_cast<gfx::VulkanTexture&>(*texture);

    // set pixels (undefined to shader-read)
    EXPECT_NO_THROW(set_pixels(staging,
                               texture.get(),
                               ramp_pixels,
                               full_image_size_bytes,
                               1,
                               kFullWidth,
                               kFullHeight,
                               kFullWidth,
                               kFullHeight));

    // get pixels
    EXPECT_NO_THROW(
        staging.getPixels(vk_texture.getImage(),
                          kFullWidth,
                          kFullHeight,
                          1,
                          PixelFormat::kRGBA8,
                          reinterpret_cast<std::byte*>(full_get_pixels.data()),
                          full_image_size_bytes));

    // compare with ramp
    EXPECT_EQ(ramp_pixels, full_get_pixels);

    // set pixels with ramp again
    EXPECT_NO_THROW(set_pixels(staging,
                               texture.get(),
                               ramp_pixels,
                               full_image_size_bytes,
                               1,
                               kFullWidth,
                               kFullHeight,
                               kFullWidth,
                               kFullHeight));

    // get pixels again
    EXPECT_NO_THROW(
        staging.getPixels(vk_texture.getImage(),
                          kFullWidth,
                          kFullHeight,
                          1,
                          PixelFormat::kRGBA8,
                          reinterpret_cast<std::byte*>(full_get_pixels.data()),
                          full_image_size_bytes));

    // compare with ramp
    EXPECT_EQ(ramp_pixels, full_get_pixels);

    // set sub-region
    EXPECT_NO_THROW(set_pixels(staging,
                               texture.get(),
                               region_ramp_pixels,
                               full_image_size_bytes,
                               1,
                               kRegionWidth,
                               kRegionHeight,
                               kFullWidth,
                               kFullHeight));

    // get sub-region
    EXPECT_NO_THROW(
        staging.getPixels(vk_texture.getImage(),
                          kRegionWidth,
                          kRegionHeight,
                          1,
                          PixelFormat::kRGBA8,
                          reinterpret_cast<std::byte*>(region_get_pixels.data()),
                          region_image_size_bytes));

    // compare
    EXPECT_EQ(region_ramp_pixels, region_get_pixels);

    // done
    EXPECT_NO_THROW(vk_resource_mgr.destroyTexture(std::move(texture)));
  }
}

TEST_F(VulkanDeviceContextTest, StagingContextArrayTest) {
  static constexpr uint32_t kFullWidth = 1024;
  static constexpr uint32_t kFullHeight = 1024;
  static constexpr uint32_t kRegionWidth = 1023;
  static constexpr uint32_t kRegionHeight = 1023;
  static constexpr uint32_t kLayerCount = 4;
  for (auto& device : this->device_contexts_) {
    EXPECT_TRUE(device->getLimits().areValid());
    auto& vk_resource_mgr =
        static_cast<VulkanResourceManager&>(device->getResourceManager());
    auto const& vk_device = static_cast<const VulkanDeviceContext&>(*device);
    auto& staging = vk_device.getStagingContext();

    // create some content
    auto ramp_pixels = make_ramp_pixels(kFullWidth, kFullHeight, kLayerCount);
    auto region_ramp_pixels = make_ramp_pixels(kRegionWidth, kRegionHeight, kLayerCount);

    // image sizes
    const uint64_t full_image_size_bytes = kFullWidth * kFullHeight * kLayerCount * 4;
    const uint64_t region_image_size_bytes =
        kRegionWidth * kRegionHeight * kLayerCount * 4;

    // prepare to get pixels
    std::vector<uint8_t> full_get_pixels(full_image_size_bytes);
    std::vector<uint8_t> region_get_pixels(region_image_size_bytes);

    // create an empty texture (usage: sampled + transfer source + transfer dest)
    resource_ptr<Texture> texture_array;
    EXPECT_NO_THROW(texture_array = vk_resource_mgr.createTexture(
                        "simple texture array",
                        kFullWidth,
                        kFullHeight,
                        kLayerCount,
                        PixelFormat::kRGBA8,
                        1,
                        true,
                        ImageUsageBits::kNone,
                        get_default_sampler_state_for_format(PixelFormat::kRGBA8),
                        nullptr));
    auto& vk_texture_array = static_cast<gfx::VulkanTexture&>(*texture_array);

    // set pixels (undefined to shader-read)
    EXPECT_NO_THROW(set_pixels(staging,
                               texture_array.get(),
                               ramp_pixels,
                               full_image_size_bytes,
                               kLayerCount,
                               kFullWidth,
                               kFullHeight,
                               kFullWidth,
                               kFullHeight));

    // get pixels
    EXPECT_NO_THROW(
        staging.getPixels(vk_texture_array.getImage(),
                          kFullWidth,
                          kFullHeight,
                          kLayerCount,
                          PixelFormat::kRGBA8,
                          reinterpret_cast<std::byte*>(full_get_pixels.data()),
                          full_image_size_bytes));

    // compare with ramp
    EXPECT_EQ(ramp_pixels, full_get_pixels);

    // set pixels with ramp again
    EXPECT_NO_THROW(set_pixels(staging,
                               texture_array.get(),
                               ramp_pixels,
                               full_image_size_bytes,
                               kLayerCount,
                               kFullWidth,
                               kFullHeight,
                               kFullWidth,
                               kFullHeight));

    // get pixels again
    EXPECT_NO_THROW(
        staging.getPixels(vk_texture_array.getImage(),
                          kFullWidth,
                          kFullHeight,
                          kLayerCount,
                          PixelFormat::kRGBA8,
                          reinterpret_cast<std::byte*>(full_get_pixels.data()),
                          full_image_size_bytes));

    // compare with ramp
    EXPECT_EQ(ramp_pixels, full_get_pixels);

    // set sub-region
    EXPECT_NO_THROW(set_pixels(staging,
                               texture_array.get(),
                               region_ramp_pixels,
                               full_image_size_bytes,
                               kLayerCount,
                               kRegionWidth,
                               kRegionHeight,
                               kFullWidth,
                               kFullHeight));

    // get sub-region
    EXPECT_NO_THROW(
        staging.getPixels(vk_texture_array.getImage(),
                          kRegionWidth,
                          kRegionHeight,
                          kLayerCount,
                          PixelFormat::kRGBA8,
                          reinterpret_cast<std::byte*>(region_get_pixels.data()),
                          region_image_size_bytes));

    // compare
    EXPECT_EQ(region_ramp_pixels, region_get_pixels);

    // done
    EXPECT_NO_THROW(vk_resource_mgr.destroyTexture(std::move(texture_array)));
  }
}

/*
 * Test getPixels from MS texture
 */

TEST_F(VulkanDeviceContextTest, TextureGetMS) {
  static constexpr uint32_t kFullWidth = 512;
  static constexpr uint32_t kFullHeight = 512;
  static constexpr PixelFormat kPixelFormat = PixelFormat::kRGBA8;
  for (auto& device : this->device_contexts_) {
    auto& resource_mgr = device->getResourceManager();

    resource_ptr<Texture> ms_texture;
    ASSERT_NO_THROW(ms_texture = resource_mgr.createTexture("Test TextureGetMS",
                                                            kFullWidth,
                                                            kFullHeight,
                                                            1,
                                                            kPixelFormat,
                                                            4,
                                                            false,
                                                            ImageUsageBits::kNone,
                                                            TextureSamplerState(),
                                                            nullptr));

    ScopeGuard assert_cleanup = [&]() {
      if (ms_texture) {
        resource_mgr.destroyTexture(std::move(ms_texture));
      }
    };

    auto black_pixels = make_black_pixels(kFullWidth, kFullHeight, 1);

    ASSERT_NO_THROW(ms_texture->clearPixels());

    std::vector<uint8_t> read_pixels(kFullWidth * kFullHeight *
                                     pixelFormatDataSize(kPixelFormat));
    ASSERT_NO_THROW(ms_texture->getPixels(kFullWidth,
                                          kFullHeight,
                                          1,
                                          kPixelFormat,
                                          read_pixels.data(),
                                          read_pixels.size()));

    ASSERT_TRUE(read_pixels == black_pixels);

    EXPECT_NO_THROW(resource_mgr.destroyTexture(std::move(ms_texture)));
  }
}

/*
 * Vulkan RenderPass
 */
TEST_F(VulkanDeviceContextTest, CanCreateVulkanRenderPassWithSubpasses) {
  static constexpr uint32_t tex_size = 32;
  static constexpr uint32_t samples = 4;
  for (auto& device : this->device_contexts_) {
    auto& resource_mgr = device->getResourceManager();

    resource_ptr<Texture> rgba_tex;
    resource_ptr<Texture> id1a_tex;
    resource_ptr<Texture> id1b_tex;
    resource_ptr<Texture> id2_tex;
    resource_ptr<Texture> depth_tex;
    resource_ptr<RenderPass> render_pass;

    ScopeGuard assert_cleanup = [&]() {
      if (render_pass) {
        resource_mgr.destroyRenderPass(std::move(render_pass));
      }
      if (rgba_tex) {
        resource_mgr.destroyTexture(std::move(rgba_tex));
      }
      if (id1a_tex) {
        resource_mgr.destroyTexture(std::move(id1a_tex));
      }
      if (id1b_tex) {
        resource_mgr.destroyTexture(std::move(id1b_tex));
      }
      if (id2_tex) {
        resource_mgr.destroyTexture(std::move(id2_tex));
      }
      if (depth_tex) {
        resource_mgr.destroyTexture(std::move(depth_tex));
      }
    };

    // Create textures to use as attachments
    auto create_texture = [&](const PixelFormat pixel_format,
                              const ImageUsageBits usage_bits) -> resource_ptr<Texture> {
      return resource_mgr.createTexture(
          "CanCreateVulkanRenderPassWithSubpasses Test",
          tex_size,
          tex_size,
          1,
          pixel_format,
          samples,
          false,
          usage_bits,
          get_default_sampler_state_for_format(pixel_format));
    };

    rgba_tex = create_texture(PixelFormat::kRGBA8, ImageUsageBits::kColorAttachmentBit);
    id1a_tex = create_texture(PixelFormat::kR32UI, ImageUsageBits::kColorAttachmentBit);
    id1b_tex = create_texture(PixelFormat::kR32UI, ImageUsageBits::kColorAttachmentBit);
    id2_tex = create_texture(PixelFormat::kR32UI, ImageUsageBits::kColorAttachmentBit);
    depth_tex =
        create_texture(PixelFormat::kDepth, ImageUsageBits::kDepthStencilAttachmentBit);

    ASSERT_NE(rgba_tex, nullptr);
    ASSERT_NE(id1a_tex, nullptr);
    ASSERT_NE(id1b_tex, nullptr);
    ASSERT_NE(id2_tex, nullptr);
    ASSERT_NE(depth_tex, nullptr);

    AttachmentManager attachment_mgr;
    attachment_mgr.setAttachment(Framebuffer::Attachment::kColor0, rgba_tex.get());
    attachment_mgr.setAttachment(Framebuffer::Attachment::kColor1, id1a_tex.get());
    attachment_mgr.setAttachment(Framebuffer::Attachment::kColor2, id1b_tex.get());
    attachment_mgr.setAttachment(Framebuffer::Attachment::kColor3, id2_tex.get());
    attachment_mgr.setAttachment(Framebuffer::Attachment::kDepth, depth_tex.get());

    // Default RenderPass - 1 automatic subpass using all attachments
    render_pass = resource_mgr.createRenderPass("Test",
                                                attachment_mgr.getLayout(),
                                                gfx::RenderPass::ClearBits::kAll,
                                                ImageLayout::kUndefined,
                                                ImageLayout::kAttachment);
    ASSERT_NE(render_pass, nullptr);
    EXPECT_NO_THROW(resource_mgr.destroyRenderPass(std::move(render_pass)));
    ASSERT_EQ(render_pass, nullptr);

    // 2 subpasses. Use first subpass attachments as inputs to second subpass
    std::vector<SubpassDescriptor> subpasses(2);
    subpasses[0].attachments = {Framebuffer::Attachment::kColor0,
                                Framebuffer::Attachment::kDepth};
    subpasses[1].attachments = {Framebuffer::Attachment::kColor1,
                                Framebuffer::Attachment::kColor2,
                                Framebuffer::Attachment::kColor3};
    subpasses[1].input_attachments = {Framebuffer::Attachment::kColor0,
                                      Framebuffer::Attachment::kDepth};
    render_pass = resource_mgr.createRenderPass("Test",
                                                attachment_mgr.getLayout(),
                                                gfx::RenderPass::ClearBits::kNone,
                                                ImageLayout::kAttachment,
                                                ImageLayout::kAttachment,
                                                subpasses);
    ASSERT_NE(render_pass, nullptr);
    EXPECT_NO_THROW(resource_mgr.destroyRenderPass(std::move(render_pass)));
    ASSERT_EQ(render_pass, nullptr);

    // Destroy textures
    EXPECT_NO_THROW(resource_mgr.destroyTexture(std::move(rgba_tex)));
    EXPECT_NO_THROW(resource_mgr.destroyTexture(std::move(id1a_tex)));
    EXPECT_NO_THROW(resource_mgr.destroyTexture(std::move(id1b_tex)));
    EXPECT_NO_THROW(resource_mgr.destroyTexture(std::move(id2_tex)));
    EXPECT_NO_THROW(resource_mgr.destroyTexture(std::move(depth_tex)));
  }
}

/*
 * Resource leak tests
 */

// Helpers for sinked_ptr tests
class SinkedPtrTestSink;
template <typename T>
using test_sinked_ptr = sinked_ptr<T, SinkedPtrTestSink>;

class SinkedPtrTestSink {
 public:
  test_sinked_ptr<int> createObject() {
    return make_sinked_ptr<int, SinkedPtrTestSink>(new int{0});
  }

  void destroyObject(test_sinked_ptr<int> ptr) { unlockPtr(ptr); }

 private:
  template <typename T>
  void unlockPtr(test_sinked_ptr<T>& p) {
    auto& deleter = p.get_deleter();
    ASSERT_EQ(typeid(deleter), typeid(SinkedDeleter<T, SinkedPtrTestSink>));
    deleter.allow_delete_ = true;
  }
};

TEST(SinkedPtrTests, SinkedPtrTest) {
  SinkedPtrTestSink sink;
  // first test unlock and delete
  {
    auto ptr = sink.createObject();
    sink.destroyObject(std::move(ptr));
  }

  LogCapture log_capture;
  static constexpr std::string_view compare_str(
      "int must be destroyed using GfxDriverTests::SinkedPtrTestSink sink");

  // Test sinked_ptr leak trap and verify correct error message
  { auto ptr = sink.createObject(); }
  EXPECT_TRUE(log_capture.contains(compare_str));
}

TYPED_TEST(TypedDeviceContextTest, ResourcePtrLeakTest) {
  constexpr uint64_t buffer_size = 1024;
  auto* device = this->device_contexts_[0].get();

  LogCapture log_capture;
  static constexpr std::string_view compare_str(
      "gfx::Buffer must be destroyed using gfx::ResourceManager sink");

  {
    // Trying to delete a resource_ptr directly, must use ResourceManager sink
    auto buffer =
        device->getResourceManager().createBaseBuffer("ResourcePtrLeakDeathTest",
                                                      {BufferType::kVertexBuffer,
                                                       buffer_size,
                                                       BufferUsageBits::kNone,
                                                       BufferAccessType::kDeviceLocal});
    buffer = nullptr;
    EXPECT_TRUE(log_capture.contains(compare_str));
  }

  // explicitly cleanup leaked resource so the test fixture doesn't fail the test
  this->driver_->destroyDeviceContext(std::move(this->device_contexts_[0]));
}

TEST(GfxHelpersTest, TestTileBuilder) {
  std::vector<Rect2D> tiles;
  std::vector<Rect2D> compare_tiles;

  // Test image smaller than tile size generates 1 image size tile
  {
    build_tile_queue(tiles, 2, 3, 10, 10);
    EXPECT_EQ(tiles.size(), 1u);
    EXPECT_EQ(tiles[0], (Rect2D{0, 0, 2u, 3u}));
  }

  // Test image is exact multiple of tile size
  {
    build_tile_queue(tiles, 30, 20, 10, 10);
    compare_tiles = {{0, 0, 10, 10},
                     {10, 0, 10, 10},
                     {20, 0, 10, 10},
                     {0, 10, 10, 10},
                     {10, 10, 10, 10},
                     {20, 10, 10, 10}};
    EXPECT_EQ(tiles.size(), compare_tiles.size());
    EXPECT_EQ(tiles, compare_tiles);
  }

  // Test partial tiles
  {
    build_tile_queue(tiles, 15, 16, 12, 11);
    compare_tiles = {{0, 0, 12, 11}, {12, 0, 3, 11}, {0, 11, 12, 5}, {12, 11, 3, 5}};
    EXPECT_EQ(tiles.size(), compare_tiles.size());
    EXPECT_EQ(tiles, compare_tiles);
  }
}

}  // namespace GfxDriverTests

int main(int argc, char* argv[]) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    testing::FLAGS_gtest_death_test_style = "threadsafe";
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  return err;
}
