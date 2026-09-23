/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifdef HAVE_CUDA
#include <cuda.h>
#endif  // HAVE_CUDA

#include "Tests/RenderTests/GfxDriver/GfxDriverTestFixtures.h"
#include "Tests/TestHelpers.h"

#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/Pipeline/Pipeline.h"
#include "GfxDriver/Pipeline/PipelineDescriptor.h"
#include "GfxDriver/Resources/BufferAllocator.h"
#include "GfxDriver/Resources/BufferLayout.h"
#include "GfxDriver/Resources/VertexBuffer.h"
#include "Tests/RenderTests/Utils/AttachmentUtils.h"
#include "Tests/RenderTests/Utils/GoldenImage.h"
#include "Tests/TestHelpers.h"

namespace GfxInteropTests {

namespace {
// Render dimensions for all tests (golden images 320x240)
#ifdef HAVE_CUDA  // avoid clang unused variable warnings
static constexpr uint32_t render_width = 320;
static constexpr uint32_t render_height = 240;

static std::string_view kFlushName{"GfxInteropTest"};
#endif

// Flag to force writing of all golden image files
static constexpr bool regenerate_golden_images = false;

// File paths
static std::string g_golden_images_path(std::string(RENDER_TESTS_PATH) +
                                        "GfxInterop/golden_images/");
static std::string g_shaders_path(std::string(RENDER_TESTS_PATH) + "GfxInterop/shaders/");
}  // namespace

//
// GfxInteropShaderLibrary
//
class GfxInteropShaderLibrary {
 public:
  static void init(Library& library) {
    library.addFromManifestFile("ShaderManifest.json", g_shaders_path);
  }
};

template <typename GTEST_BASE, typename DRIVER_TYPE, typename SHADER_LIBRARY>
class InteropTestBase
    : public GfxDriverTests::DeviceContextTest<GTEST_BASE, DRIVER_TYPE, SHADER_LIBRARY> {
  using Base = GfxDriverTests::DeviceContextTest<GTEST_BASE, DRIVER_TYPE, SHADER_LIBRARY>;

 protected:
  void SetUp() override {
    Base::SetUp();
    golden_image_ = std::make_unique<GoldenImage>(
        g_golden_images_path, this->driver_->getType(), regenerate_golden_images);

#ifdef HAVE_CUDA
    std::map<heavyai::UUID, DeviceContext*> uuid_to_device_ctx_map;
    for (auto& device_ctx : this->device_contexts_) {
      if (device_ctx->getVendor() == gfx::DeviceVendor::kNvidia) {
        ASSERT_EQ(
            uuid_to_device_ctx_map.try_emplace(device_ctx->getGpuUUID(), device_ctx.get())
                .second,
            true);
      }
    }

    ASSERT_EQ(cuInit(0), CUDA_SUCCESS);
    int device_count;
    ASSERT_EQ(cuDeviceGetCount(&device_count), CUDA_SUCCESS);
    for (int i = 0; i < device_count; ++i) {
      CUdevice cuda_device;
      ASSERT_EQ(cuDeviceGet(&cuda_device, i), CUDA_SUCCESS) << i;

      CUuuid cuda_uuid;
      ASSERT_EQ(cuDeviceGetUuid(&cuda_uuid, cuda_device), CUDA_SUCCESS) << i;

      auto itr = uuid_to_device_ctx_map.find(heavyai::UUID(cuda_uuid.bytes));
      if (itr != uuid_to_device_ctx_map.end()) {
        auto const* device_ctx = itr->second;
        ASSERT_EQ(device_ctx->getVendor(), gfx::DeviceVendor::kNvidia)
            << device_ctx->getGpuUUID() << ", " << device_ctx->getGpuId() << ", " << i;

        CUcontext cuda_ctx;
        ASSERT_EQ(cuCtxCreate(&cuda_ctx, 0, cuda_device), CUDA_SUCCESS);
        ASSERT_NE(nullptr, cuda_ctx);

        ASSERT_EQ(cuda_contexts_.try_emplace(itr->first, cuda_ctx).second, true)
            << device_ctx->getGpuUUID() << ", " << device_ctx->getGpuId() << ", " << i;

        ASSERT_EQ(cuda_device_nums_.try_emplace(itr->first, i).second, true)
            << device_ctx->getGpuUUID() << ", " << device_ctx->getGpuId() << ", " << i;

        uuid_to_device_ctx_map.erase(itr);
      }
    }

    // clear out nvidia device contexts that may have been hidden, i.e. via
    // CUDA_VISIBLE_DEVICES, but keep non-nvidia device contexts.
    for (auto itr = std::find_if(this->device_contexts_.begin(),
                                 this->device_contexts_.end(),
                                 [&uuid_to_device_ctx_map](auto const& device_context) {
                                   return uuid_to_device_ctx_map.find(
                                              device_context->getGpuUUID()) !=
                                          uuid_to_device_ctx_map.end();
                                 });
         itr != this->device_contexts_.end();) {
      this->driver_->destroyDeviceContext(std::move(*itr));
      this->device_contexts_.erase(itr);
    }
#endif  // HAVE_CUDA
  }

  void TearDown() override {
#ifdef HAVE_CUDA
    for (auto& [uuid, cuda_ctx] : cuda_contexts_) {
      ASSERT_EQ(cuCtxDestroy(cuda_ctx), CUDA_SUCCESS) << uuid;
    }
    cuda_contexts_.clear();
    cuda_device_nums_.clear();
#endif  // HAVE_CUDA

    Base::TearDown();
  }

  void setExternalApiContext(const gfx::DeviceContext& device_ctx) {
#ifdef HAVE_CUDA
    auto itr = cuda_contexts_.find(device_ctx.getGpuUUID());
    if (itr != cuda_contexts_.end()) {
      ASSERT_EQ(cuCtxSetCurrent(itr->second), CUDA_SUCCESS) << itr->first;
    }
#endif  // HAVE_CUDA
  }

  int getDeviceNum(const gfx::DeviceContext& device_ctx) {
#ifdef HAVE_CUDA
    auto itr = cuda_device_nums_.find(device_ctx.getGpuUUID());
    if (itr != cuda_device_nums_.end()) {
      return itr->second;
    }
#endif  // HAVE_CUDA
    return 0;
  }

#ifdef HAVE_CUDA

#define ROUND_UP_TO_GRANULARITY(s, g) (((s + (g - 1)) / g) * g)

  using SetCUDAContextCallback = std::function<void(const gfx::DeviceContext&)>;
  using GetDeviceNumCallback = std::function<int(const gfx::DeviceContext&)>;

  static void checkError(CUresult status) {
    if (status != CUDA_SUCCESS) {
      throw std::runtime_error("CUDA Error: " + std::to_string(status));
    }
  }

  static size_t getGranularity(const DeviceId device_num) {
    CUmemAllocationProp allocation_prop{};
    allocation_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    allocation_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    allocation_prop.location.id = device_num;
    size_t granularity{};
    checkError(cuMemGetAllocationGranularity(
        &granularity, &allocation_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    return granularity;
  }

  struct MemoryAllocation {
    uint64_t device_ptr;
    size_t padded_size;
    int fd;
  };

  static MemoryAllocation allocateCudaMemory(const size_t num_bytes,
                                             const size_t granularity,
                                             const DeviceId device_num) {
    // round up size
    auto const padded_num_bytes = ROUND_UP_TO_GRANULARITY(num_bytes, granularity);

    // reserve memory
    CUdeviceptr device_ptr{};
    checkError(cuMemAddressReserve(&device_ptr, padded_num_bytes, granularity, 0, 0));

    // create an allocation handle
    CUmemAllocationProp allocation_prop{};
    allocation_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    allocation_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    allocation_prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    allocation_prop.location.id = device_num;
    CUmemGenericAllocationHandle handle{};
    checkError(cuMemCreate(&handle, padded_num_bytes, &allocation_prop, 0));

    // export that handle to a POSIX file descriptor
    int fd{-1};
    checkError(cuMemExportToShareableHandle(
        &fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));

    // map the memory
    checkError(cuMemMap(device_ptr, padded_num_bytes, 0, handle, 0));

    // release the handle
    // memory will stay allocated until unmapped
    checkError(cuMemRelease(handle));

    // set the allocated memory range to read/write
    CUmemAccessDesc access_desc{};
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id = device_num;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    checkError(cuMemSetAccess(device_ptr, padded_num_bytes, &access_desc, 1));

    // return allocation info
    return {static_cast<uint64_t>(device_ptr), padded_num_bytes, fd};
  }

  static void releaseCudaMemory(const MemoryAllocation& allocation) {
    // close shareable handle
    close(allocation.fd);

    // unmap and release CUDA memory
    checkError(cuMemUnmap(allocation.device_ptr, allocation.padded_size));
    checkError(cuMemAddressFree(allocation.device_ptr, allocation.padded_size));
  }

#endif  // HAVE_CUDA

#ifdef HAVE_CUDA
  std::map<heavyai::UUID, CUcontext> cuda_contexts_;
  std::map<heavyai::UUID, int> cuda_device_nums_;
#endif  // HAVE_CUDA
  std::unique_ptr<GoldenImage> golden_image_;
};

#define USE_TYPED_TEST 0

#if USE_TYPED_TEST
template <typename DRIVER_TYPE>
using TypedInteropTest =
    InteropTestBase<::testing::Test, DRIVER_TYPE, GfxInteropShaderLibrary>;
TYPED_TEST_SUITE(TypedInteropTest, GfxDriverTests::DriverTypes);
#else
using InteropTest =
    InteropTestBase<::testing::Test, VulkanDriver, GfxInteropShaderLibrary>;
#endif

//
// Import CUDA Buffer
//

#ifdef HAVE_CUDA

struct ImportCudaBufferTestExecutor : public InteropTest {
  static void executeTest(gfx::GfxContext& gfx_context,
                          DeviceContextVector& device_contexts,
                          GoldenImage& golden_image,
                          SetCUDAContextCallback set_cuda_context_callback,
                          GetDeviceNumCallback get_device_num_callback) {
    // Generate spirv
    auto caches = gfx_context.getShaderManager().createCacheVectorFromTemplate(
        {{"RenderTests/importCudaBufferTest.vert"},
         {"RenderTests/importCudaBufferTest.frag"}});
    ASSERT_NE(caches.size(), size_t{0});
    ASSERT_NE(caches[0]->getSpirv().size(), size_t{0});
    ASSERT_NE(caches[1]->getSpirv().size(), size_t{0});

    for (auto& device : device_contexts) {
      // begin_renderdoc_vulkan_capture();

      set_cuda_context_callback(*device);

      auto device_num = get_device_num_callback(*device);

      //
      // CUDA Stuff
      // build a dummy chunk with ten POINT values
      // and a dummy QOB with ten rows with the following format
      //
      // uint64_t key
      // uint64_t point_ptr
      // uint64_t point_num
      // double a
      // double b
      //
      // this is what you'd get if you did a render query like this
      //
      // SELECT p, a, b FROM points;
      //

      auto const granularity = getGranularity(device_num);

      struct PointType {
        double x;
        double y;
      };

      struct RowType {
        int64_t key;
        uint64_t point_ptr;
        int64_t point_num;
        double a;
        double b;
      };

      static constexpr uint32_t kNumRows = 10u;

      static constexpr std::array<PointType, kNumRows> chunk_values = {
          PointType{0.1, 0.2},
          PointType{0.3, 0.4},
          PointType{0.1, 0.2},
          PointType{0.3, 0.4},
          PointType{0.1, 0.2},
          PointType{0.3, 0.4},
          PointType{0.1, 0.2},
          PointType{0.3, 0.4},
          PointType{0.1, 0.2},
          PointType{0.3, 0.4}};

      auto const chunk =
          allocateCudaMemory(sizeof(chunk_values), granularity, device_num);

      checkError(cuMemcpyHtoD(chunk.device_ptr, &chunk_values, sizeof(chunk_values)));

      std::array<RowType, kNumRows> qob_values;
      for (uint32_t i = 0u; i < kNumRows; i++) {
        qob_values[i].key = i;
        qob_values[i].point_ptr = chunk.device_ptr + (i * sizeof(PointType));
        qob_values[i].point_num = 2;
        qob_values[i].a = 42.0;
        qob_values[i].b = 17.0;
      }

      auto const qob = allocateCudaMemory(sizeof(qob_values), granularity, device_num);

      checkError(cuMemcpyHtoD(qob.device_ptr, &qob_values, sizeof(qob_values)));

      //
      // Vulkan/Renderer stuff
      // mostly from GfxDriverTest drawOneTriangle
      //

      auto& resource_mgr = device->getResourceManager();

      // material
      auto material = resource_mgr.createMaterial("Test", caches);
      ASSERT_NE(material, nullptr);

      // attachment manager and textures
      auto [attachment_mgr, textures] =
          build_attachments(resource_mgr,
                            render_width,
                            render_height,
                            {{PixelFormat::kRGBA8, Framebuffer::Attachment::kColor0}});

      // renderpass and framebuffer
      auto render_pass = resource_mgr.createRenderPass("Test",
                                                       attachment_mgr.getLayout(),
                                                       gfx::RenderPass::ClearBits::kAll,
                                                       ImageLayout::kUndefined,
                                                       ImageLayout::kAttachment);
      ASSERT_NE(render_pass, nullptr);
      auto framebuffer = resource_mgr.createFramebuffer(
          "Test", *render_pass, attachment_mgr, render_width, render_height, 1u);
      ASSERT_NE(framebuffer, nullptr);

      // vertex buffer, layout, and attr map
      auto buffer_layout = std::make_shared<InterleavedBufferLayout>();
      ASSERT_NE(buffer_layout, nullptr);
      buffer_layout->addAttribute("in_position", BufferAttrType::kVec2f);
      buffer_layout->addAttribute("in_color", BufferAttrType::kVec3f);

      // clang-format off
      std::array<float, 15> vertex_data{-0.5f,  0.5f,  1.0f, 0.0f, 0.0f,
                                         0.0f, -0.5f,  0.0f, 1.0f, 0.0f,
                                         0.5f,  0.5f,  0.0f, 0.0f, 1.0f};
      // clang-format on

      auto vertex_buffer_wrapper =
          resource_mgr.createBuffer("Test",
                                    {BufferType::kVertexBuffer,
                                     vertex_data.size() * sizeof(float),
                                     BufferUsageBits::kLayoutBufferBit});
      ASSERT_NE(vertex_buffer_wrapper, nullptr);

      auto* vertex_buffer = static_cast<VertexBuffer*>(vertex_buffer_wrapper.get());
      vertex_buffer->updateSubDataWithLayout(
          vertex_data.data(), vertex_buffer_wrapper->getNumBytes(), 0, buffer_layout);

      PrimitiveAssemblyAttrInfo attr_info{
          {vertex_buffer, buffer_layout},
          {{"in_position", "in_position"}, {"in_color", "in_color"}}};

      // primitive assembly
      auto primitive_assembly = resource_mgr.createPrimitiveAssembly(
          "Test", PrimitiveTopology::kTriangleList, *material, attr_info);

      struct PushConstants {
        uint64_t chunk_cuda;
        uint64_t chunk_vulkan;
        uint64_t qob_vulkan;
      };

      // pipeline descriptor (push constants are the only addition)
      PipelineDescriptor pipeline_desc;
      gfx::PushConstantRange pcr(gfx::ShaderStageBits::kVertex, 0, sizeof(PushConstants));
      pipeline_desc.setPushConstantRanges({pcr});

      // pipeline
      auto pipeline = resource_mgr.createGraphicsPipeline(
          "Test", *material, pipeline_desc, primitive_assembly.get());
      pipeline->create(*render_pass);

      // wrap CUDA buffers with Vulkan buffers
      auto chunk_wrapper =
          resource_mgr.createBuffer("Chunk Wrapper",
                                    {gfx::BufferType::kUnspecified,
                                     chunk.padded_size,
                                     gfx::BufferUsageBits::kStorageBufferBit |
                                         gfx::BufferUsageBits::kDeviceAddressBit,
                                     gfx::BufferAccessType::kDeviceLocal,
                                     chunk.fd});
      auto qob_wrapper =
          resource_mgr.createBuffer("QOB Wrapper",
                                    {gfx::BufferType::kUnspecified,
                                     qob.padded_size,
                                     gfx::BufferUsageBits::kStorageBufferBit |
                                         gfx::BufferUsageBits::kDeviceAddressBit,
                                     gfx::BufferAccessType::kDeviceLocal,
                                     qob.fd});

      // build push constants
      PushConstants push_constants;
      push_constants.chunk_cuda = chunk.device_ptr;
      push_constants.chunk_vulkan = chunk_wrapper->getBuffer().getDeviceAddress();
      push_constants.qob_vulkan = qob_wrapper->getBuffer().getDeviceAddress();

      // render via command list
      device->getCommandExecutor().setDefaultViewportAndRenderArea(
          0, 0, render_width, render_height);
      auto& cmd_list = device->getCommandList();
      cmd_list.beginRenderPass(*render_pass, *framebuffer)
          .setPushConstants(*pipeline,
                            "Buffer Addresses",
                            gfx::ShaderStageBits::kVertex,
                            &push_constants,
                            sizeof(PushConstants))
          .drawVertices(*pipeline, *vertex_buffer, 3)
          .endRenderPass()
          .flush(kFlushName);

      // Image operation (golden-image compare)
      // this will pass as long as the row_index in the shader is even
      EXPECT_TRUE(golden_image.compare(
          *framebuffer, Framebuffer::Attachment::kColor0, "import_cuda_buffer_test"));

      // destroy these first
      resource_mgr.destroyBuffer(std::move(chunk_wrapper));
      resource_mgr.destroyBuffer(std::move(qob_wrapper));

      // then do this
      releaseCudaMemory(chunk);
      releaseCudaMemory(qob);

      // then destroy remaining resources
      resource_mgr.destroyBuffer(std::move(vertex_buffer_wrapper));
      resource_mgr.destroyPipeline(std::move(pipeline));
      material = nullptr;
      resource_mgr.destroyFramebuffer(std::move(framebuffer));
      for (auto& texture : textures) {
        resource_mgr.destroyTexture(std::move(texture));
      }
      resource_mgr.destroyRenderPass(std::move(render_pass));
      // end_renderdoc_vulkan_capture();
    }
  }
};

// cuda is currently the only supported external api
#if USE_TYPED_TEST
TYPED_TEST(TypedInteropTest, ImportCudaBufferTest) {
#else
TEST_F(InteropTest, ImportCudaBufferTest) {
#endif
  ImportCudaBufferTestExecutor::executeTest(
      *this->gfx_context_,
      this->device_contexts_,
      *this->golden_image_,
      [this](auto const& device) { this->setExternalApiContext(device); },
      [this](auto const& device) -> int { return this->getDeviceNum(device); });
}

#endif  // HAVE_CUDA

//
// Slab-Allocated Buffer
//

#ifdef HAVE_CUDA

struct SlabAllocatedBufferTestExecutor : public InteropTest {
  class DummySlabAllocator : public gfx::BufferAllocator {
   public:
    DummySlabAllocator(const gfx::DeviceContext& device_ctx,
                       Buffer& wrapper_buffer,
                       const uint64_t slab_base)
        : device_ctx_{device_ctx}
        , wrapper_buffer_{wrapper_buffer}
        , slab_base_{slab_base}
        , allocation_base_{0ULL} {}
    ~DummySlabAllocator() override = default;

    BufferAllocationUqPtr alloc(const uint64_t num_bytes) final {
      auto allocation = std::make_unique<gfx::BufferAllocation>(
          wrapper_buffer_, slab_base_, num_bytes, allocation_base_);
      CHECK(allocation);
      allocation_base_ += num_bytes;
      return allocation;
    }

    void free(BufferAllocationUqPtr allocation) final {
      CHECK(allocation);
      CHECK_GE(allocation_base_, allocation->num_bytes);
      allocation_base_ -= allocation->num_bytes;
      allocation = nullptr;
    }

    uint64_t getAllocationBase() const { return allocation_base_; }

    void validateCreateInfo(const gfx::BufferCreateInfo&) final {}

    const gfx::DeviceContext& getDeviceContext() const override { return device_ctx_; }

   private:
    const gfx::DeviceContext& device_ctx_;
    Buffer& wrapper_buffer_;
    const uint64_t slab_base_;
    uint64_t allocation_base_;
  };

  static void executeTest(gfx::GfxContext& gfx_context,
                          DeviceContextVector& device_contexts,
                          SetCUDAContextCallback set_cuda_context_callback,
                          GetDeviceNumCallback get_device_num_callback) {
    for (auto& device : device_contexts) {
      //
      // init for this device
      //

      set_cuda_context_callback(*device);
      auto device_num = get_device_num_callback(*device);

      auto& resource_mgr = device->getResourceManager();

      //
      // the slab
      //

      static constexpr uint64_t kSlabSize = 1024 * 1024 * 1024;

      auto const slab =
          allocateCudaMemory(kSlabSize, getGranularity(device_num), device_num);
      CHECK(slab.device_ptr);

      //
      // slab wrapper buffer
      //

      auto wrapper_buffer =
          resource_mgr.createBuffer("Slab Wrapper",
                                    {gfx::BufferType::kSlabWrapperBuffer,
                                     kSlabSize,
                                     gfx::BufferUsageBits::kDeviceAddressBit,
                                     gfx::BufferAccessType::kDeviceLocal,
                                     slab.fd});
      CHECK(wrapper_buffer);

      //
      // dummy slab allocator
      //

      auto dummy_slab_allocator = std::make_shared<DummySlabAllocator>(
          *device, wrapper_buffer->getBuffer(), slab.device_ptr);

      //
      // allocate buffers
      //

      static constexpr uint64_t kBufferSize = 1024 * 1024;

      auto slab_allocated_buffer_1 =
          resource_mgr.createBuffer("Slab-Allocated Buffer 1",
                                    {gfx::BufferType::kUnspecified,
                                     kBufferSize,
                                     gfx::BufferUsageBits::kStorageBufferBit},
                                    dummy_slab_allocator);
      CHECK(slab_allocated_buffer_1);

      CHECK_EQ(slab_allocated_buffer_1->getAllocationBasePtr(), slab.device_ptr);
      CHECK_EQ(slab_allocated_buffer_1->getAllocationOffsetBytes(), 0ULL);
      CHECK_EQ(slab_allocated_buffer_1->getNumBytes(), kBufferSize);

      auto slab_allocated_buffer_2 =
          resource_mgr.createBuffer("Slab-Allocated Buffer 2",
                                    {gfx::BufferType::kUnspecified,
                                     kBufferSize,
                                     gfx::BufferUsageBits::kStorageBufferBit},
                                    dummy_slab_allocator);
      CHECK(slab_allocated_buffer_2);

      CHECK_EQ(slab_allocated_buffer_2->getAllocationBasePtr(), slab.device_ptr);
      CHECK_EQ(slab_allocated_buffer_2->getAllocationOffsetBytes(), kBufferSize);
      CHECK_EQ(slab_allocated_buffer_2->getNumBytes(), kBufferSize);

      //
      // destroy buffers
      //

      resource_mgr.destroyBuffer(std::move(slab_allocated_buffer_2));

      CHECK_EQ(dummy_slab_allocator->getAllocationBase(), kBufferSize);

      resource_mgr.destroyBuffer(std::move(slab_allocated_buffer_1));

      CHECK_EQ(dummy_slab_allocator->getAllocationBase(), 0ULL);

      //
      // destroy the buffer allocator
      //

      CHECK_EQ(dummy_slab_allocator.use_count(), 1);
      dummy_slab_allocator = nullptr;

      //
      // destroy slab wrapper buffer
      //

      resource_mgr.destroyBuffer(std::move(wrapper_buffer));

      //
      // free CUDA memory
      //

      releaseCudaMemory(slab);
    }
  }
};

// cuda is currently the only supported external api
#if USE_TYPED_TEST
TYPED_TEST(TypedInteropTest, SlabAllocatedBufferTest) {
#else
TEST_F(InteropTest, SlabAllocatedBufferTest) {
#endif
  SlabAllocatedBufferTestExecutor::executeTest(
      *this->gfx_context_,
      this->device_contexts_,
      [this](auto const& device) { this->setExternalApiContext(device); },
      [this](auto const& device) -> int { return this->getDeviceNum(device); });
}

#endif  // HAVE_CUDA

}  // namespace GfxInteropTests

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
