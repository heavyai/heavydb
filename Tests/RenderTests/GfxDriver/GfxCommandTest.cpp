/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Tests/RenderTests/GfxDriver/GfxDriverTestFixtures.h"

#include "GfxDriver/Commands/CommandExecutor.h"
#include "GfxDriver/Commands/CommandList.h"
#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/Pipeline/Pipeline.h"
#include "GfxDriver/Resources/AttachmentManager.h"
#include "GfxDriver/Resources/Framebuffer.h"
#include "GfxDriver/Resources/RenderPass.h"
#include "GfxDriver/Resources/ResourcePtr.h"
#include "GfxDriver/Resources/Texture.h"
#include "GfxDriver/Resources/VertexBuffer.h"

namespace GfxDriverTests {

//
// Resource mock stubs to use when populating commands
//

//
// RenderPass
//
class RenderPassMock : public RenderPass {
 public:
  explicit RenderPassMock(const DeviceContext& device_ctx)
      : RenderPass(device_ctx, "Simple Mock RenderPass") {}
  uint32_t getNumSubpasses() const override { return 0u; }
  ResourceHandle getResourceHandle() const override { return 0u; }

 private:
  void cleanupResourceBase() override {}
  void makeEmpty() override {}
};

//
// Framebuffer
//
class FramebufferMock : public Framebuffer {
 public:
  explicit FramebufferMock(const DeviceContext& device_ctx,
                           const RenderPass& render_pass,
                           AttachmentManager& attachment_mgr)
      : Framebuffer(device_ctx,
                    "Simple Mock Framebuffer",
                    render_pass,
                    attachment_mgr,
                    16,
                    16,
                    1) {}
  ResourceHandle getResourceHandle() const override { return 0u; }

  void readPixels(const Attachment attachment,
                  const uint32_t start_x,
                  const uint32_t start_y,
                  const uint32_t width,
                  const uint32_t height,
                  const PixelFormat pixel_format,
                  void* data) override {}

  void copyToFramebuffer(Framebuffer& dst_fbo,
                         const Attachment src_attachment,
                         const uint32_t src_x,
                         const uint32_t src_y,
                         const uint32_t src_width,
                         const uint32_t src_height,
                         const Attachment dst_attachment,
                         const uint32_t dst_x,
                         const uint32_t dst_y,
                         const uint32_t dst_width,
                         const uint32_t dst_height,
                         const SamplerFilterMode filter) override {}

  void copyToFramebuffer(Framebuffer& dst_fbo,
                         const std::vector<Attachment>& attachments,
                         const uint32_t src_x,
                         const uint32_t src_y,
                         const uint32_t src_width,
                         const uint32_t src_height,
                         const uint32_t dst_x,
                         const uint32_t dst_y,
                         const uint32_t dst_width,
                         const uint32_t dst_height,
                         const bool do_async_copy) override {}

  void copyToPixelBuffer(PixelBuffer2d& dst_pbo,
                         const Attachment attachment,
                         const uint32_t start_x,
                         const uint32_t start_y,
                         const uint32_t width,
                         const uint32_t height,
                         const uint64_t offset_bytes,
                         const PixelFormat pixel_format) override {}

  void resize(const uint32_t width, const uint32_t height) override {}

  void activateEnabledAttachmentsForDrawing() override {}

 protected:
  void initResource() override {}

 private:
  void cleanupResourceBase() override {}
  void makeEmpty() override {}
};

//
// Material
//
class MaterialMock : public Material {
 public:
  explicit MaterialMock(const DeviceContext& device_ctx,
                        ShaderCacheShPtrVector& caches,
                        bool allow_duplicate_shader_stages)
      : Material(device_ctx,
                 "Simple Mock Material",
                 caches,
                 allow_duplicate_shader_stages) {}
  ~MaterialMock() override = default;

  void setSamplerAttribute(std::string_view attr_name,
                           const Texture& texture,
                           const uint32_t view_id) override {}
  void setSamplerArrayAttribute(
      std::string_view attr_name,
      const std::vector<resource_ptr<Texture>>& textures) override {}
  void setImageLoadStoreAttribute(std::string_view attr_name,
                                  const Texture& texture,
                                  const uint32_t view_id) override {}
  void setImageLoadStoreArrayAttribute(
      std::string_view attr_name,
      const std::vector<resource_ptr<Texture>>& textures) override {}
  void bindShaderStorageBufferToBlock(std::string_view block_name,
                                      const BufferWrapper& ssbo,
                                      std::optional<uint64_t> offset,
                                      std::optional<uint64_t> range) override {}
  void bindExternalUniformBufferToBlock(std::string_view block_name,
                                        const BufferWrapper& ubo) override {}
  void setAccelerationStructureAttribute(
      std::string_view attr_name,
      const AccelerationStructure& accel_structure) override{};
};

//
// Pipeline
//
class PipelineMock : public Pipeline {
 public:
  explicit PipelineMock(const DeviceContext& device_ctx, Material& material)
      : Pipeline(device_ctx, "Simple Mock Pipeline", material) {}
  ~PipelineMock() override {}

  ResourceHandle getResourceHandle() const override { return 0u; }
  ResourceHandle getPipelineHandle(uint32_t specialization_id) const override {
    return 0u;
  }
  ResourceHandle getLayout() const override { return 0u; }

  Type getType() const override { return Type::kGraphics; }

 private:
  void cleanupResourceBase() override {}
  void makeEmpty() override {}
};

//
// Texture
//
class TextureMock : public Texture {
 public:
  explicit TextureMock(const DeviceContext& device_ctx)
      : Texture(device_ctx,
                "Simple Mock Texture",
                16,
                16,
                1,
                PixelFormat::kRGBA8,
                1,
                false) {}
  ~TextureMock() override {}

  ResourceHandle getResourceHandle() const override { return 0u; }
  uint64_t getGpuAllocationSize() const override { return 0ull; }

  void resize(const uint32_t width, const uint32_t height, uint32_t depth) override {}
  void clearPixels() override {}
  void clearPixelsToValue(const ClearTextureValue& value) override {}
  void setPixels(const uint32_t width,
                 const uint32_t height,
                 const uint32_t depth,
                 const PixelFormat pixel_format,
                 const void* pixel_data) override {}
  void getPixels(const uint32_t width,
                 const uint32_t height,
                 const uint32_t depth,
                 const PixelFormat pixel_format,
                 void* pixel_data,
                 const uint64_t buffer_size) const override {}

  void initResource(const void* pixel_data) override {}

  ViewCreateResult createView(const uint32_t view_id,
                              const PixelFormat pixel_format) override {
    return {0u, false};
  }
  bool destroyView(const uint32_t view_id) override { return false; }
  bool hasView(const uint32_t view_id) const override { return false; }
  ResourceHandle getViewHandle(const uint32_t view_id) const override { return 0u; }
  PixelFormat getViewPixelFormat(const uint32_t view_id) const override {
    return PixelFormat::kCOUNT;
  }

 private:
  void cleanupResourceBase() override {}
  void makeEmpty() override {}
};

//
// Buffer
//
class BufferMock : public Buffer {
 public:
  BufferMock(const DeviceContext& device_ctx)
      : Buffer(device_ctx, "Simple Mock Buffer", {}) {}
  ~BufferMock() override = default;
  void create(const void* data,
              uint64_t num_bytes,
              std::optional<LoggingCallback> oom_logging_cb = std::nullopt) override {}
  void rebuild(const void* data,
               uint64_t num_bytes,
               std::optional<LoggingCallback> oom_logging_cb = std::nullopt) override {}

  void updateSubData(const void* data,
                     uint64_t num_bytes,
                     uint64_t byte_offset) override {}

  void getData(void* data,
               const uint64_t num_bytes,
               const uint64_t byte_offset = 0ULL) override {}

  // returns 0 if invalid
  DeviceAddress getDeviceAddress() const override { return 0; }

 private:
  void cleanupResourceBase() override {}
  void makeEmpty() override {}
};

//
// BufferWrapper
//
/*
class BufferWrapperMock : public BufferWrapper {
 public:
  explicit BufferWrapperMock(resource_ptr<Buffer> buffer)
      : BufferWrapper(std::move(buffer)) {}
  ~BufferWrapperMock() override = default;

  void create(const void* data,
              uint64_t num_bytes,
              std::optional<LoggingCallback>) override {}
  void markDirty() override {}
};
*/
//
// CommandListBuilder
// Helper class to build CommandLists containing all current commands
//
class CommandListBuilder {
 public:
  explicit CommandListBuilder(const DeviceContext& device_ctx)
      : render_pass_{std::make_unique<RenderPassMock>(device_ctx)}
      , framebuffer_{std::make_unique<FramebufferMock>(device_ctx,
                                                       *render_pass_,
                                                       attachment_mgr_)}
      , texture_{std::make_unique<TextureMock>(device_ctx)} {
    auto shader_cache = std::make_shared<ShaderCache>(spirv_t(),
                                                      std::string(),
                                                      ShaderReflection(),
                                                      ShaderStage::kVertex,
                                                      std::string(),
                                                      "Cache Test",
                                                      std::set<std::string_view>(),
                                                      0);
    auto shader_cache_vector = ShaderCacheShPtrVector{shader_cache};
    material_ = std::make_unique<MaterialMock>(device_ctx, shader_cache_vector, false);
    pipeline_ = std::make_unique<PipelineMock>(device_ctx, *material_);
    // buffer_ = std::make_unique<BufferMock>(device_ctx);
    // auto buffer_resource_ptr = make_resource_ptr<Buffer>(buffer_.get());
    // buffer_wrapper_ =
    // std::make_unique<BufferWrapperMock>(std::move(buffer_resource_ptr));
  }

  static constexpr std::string_view attr_name = "TestAttr";
  // static constexpr int32_t cmd_count = 6;
  static constexpr int32_t cmd_count = 5;
  void populateCommandList(CommandList& cmd_list) {
    cmd_list.beginRenderPass(*render_pass_, *framebuffer_);
    cmd_list.setViewport(0, 0, 0, 0);
    cmd_list.drawFullscreen(*pipeline_);
    // This is not valid, but the vertex_buffer is not accessed by the mock executor
    // so we can get away with it
    // auto* vertex_buffer = reinterpret_cast<VertexBuffer*>(buffer_wrapper_.get());
    // cmd_list.drawVertices(*pipeline_, *vertex_buffer, 1);
    cmd_list.nextSubpass();
    cmd_list.endRenderPass();
  }

 private:
  AttachmentManager attachment_mgr_;
  std::unique_ptr<RenderPassMock> render_pass_;
  std::unique_ptr<FramebufferMock> framebuffer_;
  std::unique_ptr<MaterialMock> material_;
  std::unique_ptr<PipelineMock> pipeline_;
  std::unique_ptr<TextureMock> texture_;
  // std::unique_ptr<BufferMock> buffer_;
  // std::unique_ptr<BufferWrapperMock> buffer_wrapper_;
};

//
// TestException
// used to test propagation
//
struct TestException : public std::exception {
  const char* what() const throw() override { return "Test Exception"; }
};

//
// CommandExecutor mock
// For testing CommandList execution
// Can optionally be set to throw an exception during command processing
// to test exception handling by CommandList
//
class ExecutorMock : public CommandExecutor {
 public:
  explicit ExecutorMock(const DeviceContext& device_ctx) : device_ctx_{device_ctx} {}
  ~ExecutorMock() override = default;

  bool do_throw{false};

  void maybe_throw() {
    if (do_throw) {
      throw(TestException());
    }
  }

  // CommandExecutor
  const DeviceContext& getDeviceContext() const override { return device_ctx_; }
  void setRenderArea(uint32_t x, uint32_t y, uint32_t width, uint32_t height) override {}
  void setDefaultViewportAndRenderArea(uint32_t x,
                                       uint32_t y,
                                       uint32_t width,
                                       uint32_t height) override {}
  void resetState(bool abort_on_exception) override {}

  // Command sequence
  void beginCommandSequence() override {}
  void submitCommandSequence(
      std::string_view name,
      CommandList::SubmitType submit_type,
      const std::vector<SemaphoreHandle>& wait_semaphores,
      const std::vector<SemaphoreHandle>& signal_semaphores) override {}
  void submitCommandBatch() override {}
  void waitForCompletion(bool is_profiling_run) override {}

  void beginRenderPass(RenderPass& render_pass, Framebuffer& framebuffer) override {
    maybe_throw();
  }
  void endRenderPass() override { maybe_throw(); }
  void nextSubpass() override { maybe_throw(); }

  void setViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height) override {
    maybe_throw();
  }

  void setScissor(int32_t x, int32_t y, uint32_t width, uint32_t height) override {
    maybe_throw();
  }

  void drawFullScreenQuad(Pipeline& pipeline) override { maybe_throw(); }

  void drawVertices(Pipeline& pipeline,
                    const VertexBuffer& vertex_buffer,
                    uint32_t vertex_count,
                    uint64_t offset_bytes,
                    uint32_t first_vertex,
                    uint32_t instance_count,
                    uint32_t start_instance) override {
    maybe_throw();
  }

  void drawVertices(Pipeline& pipeline,
                    const VertexBufferRefs& vertex_buffer_refs,
                    uint32_t vertex_count,
                    uint32_t first_vertex,
                    uint32_t instance_count,
                    uint32_t start_instance) override {
    maybe_throw();
  }

  void drawIndirect(Pipeline& pipeline,
                    const VertexBuffer& vertex_buffer,
                    const IndirectDrawVertexBuffer& indirect_buffer,
                    uint32_t draw_count,
                    uint32_t first_index) override {
    maybe_throw();
  }

  void drawIndirectIndexed(Pipeline& pipeline,
                           const VertexBuffer& vertex_buffer,
                           const IndexBuffer& index_buffer,
                           const IndirectDrawIndexBuffer& indirect_buffer,
                           uint32_t draw_count,
                           uint32_t first_index) override {
    maybe_throw();
  }

  void drawIndexed(Pipeline& pipeline,
                   const VertexBuffer& vertex_buffer,
                   const IndexBuffer& index_buffer,
                   uint32_t index_count,
                   uint32_t start_index,
                   int32_t vertex_offset,
                   uint32_t instance_count,
                   uint32_t start_instance) override {
    maybe_throw();
  }

  void drawIndexed(Pipeline& pipeline,
                   const VertexBufferRefs& vertex_buffer_refs,
                   const IndexBuffer& index_buffer,
                   uint32_t index_count,
                   uint32_t start_index,
                   uint32_t instance_count,
                   uint32_t start_instance) override {
    maybe_throw();
  }

  void drawMeshTasks(Pipeline& pipeline,
                     uint32_t group_count_x,
                     uint32_t group_count_y,
                     uint32_t group_count_z) override {
    maybe_throw();
  }

  void fillBuffer(const Buffer& buffer,
                  uint32_t data,
                  uint64_t num_bytes,
                  uint64_t offset) override {
    maybe_throw();
  }

  void copyBuffer(const Buffer& src_buffer,
                  const Buffer& dst_buffer,
                  uint64_t num_bytes,
                  uint64_t src_offset,
                  uint64_t dst_offset) override {
    maybe_throw();
  }

  void transitionFramebufferLayout(Framebuffer& framebuffer,
                                   ImageLayout layout) override {
    maybe_throw();
  }

  void clearFramebufferAttachment(Framebuffer& framebuffer,
                                  Framebuffer::Attachment attachment) override {
    maybe_throw();
  }

  void clearTexture(Texture& texture, ImageLayout final_layout) override {
    maybe_throw();
  }

  void clearTextureToValue(Texture& texture,
                           ClearTextureValue value,
                           ImageLayout final_layout) override {
    maybe_throw();
  }

  void imageMemoryBarrier(Texture& texture,
                          ImageMemoryBarrierType barrier_type,
                          std::optional<ImageLayout> to_layout) override {
    maybe_throw();
  }

  void bufferMemoryBarrier(const Buffer& buffer,
                           BufferMemoryBarrierType barrier_type) override {
    maybe_throw();
  }

  void setPushConstantUInt32(Pipeline& pipeline,
                             std::string_view name,
                             ShaderStageBits shader_stages,
                             uint32_t value,
                             uint32_t offset) override {
    maybe_throw();
  }

  void setPushConstants(Pipeline& pipeline,
                        std::string_view name,
                        ShaderStageBits shader_stages,
                        const void* values,
                        uint32_t num_bytes,
                        uint32_t offset) override {
    maybe_throw();
  }

  void dispatchCompute(Pipeline& pipeline,
                       uint32_t specialization_id,
                       uint32_t group_count_x,
                       uint32_t group_count_y,
                       uint32_t group_count_z) override {
    maybe_throw();
  }

  void buildAccelerationStructure(const void* build_geometry_info,
                                  const void* build_ranges_info) override {
    maybe_throw();
  }

  void traceRays(RaytracingPipeline& pipeline,
                 const ShaderBindingTable& sbt,
                 uint32_t width,
                 uint32_t height,
                 uint32_t depth) override {
    maybe_throw();
  }

  void insertLabel(const std::string_view name) override { maybe_throw(); }
  void pushLabel(const std::string_view name) override { maybe_throw(); }
  void popLabel() override { maybe_throw(); }

  void resetQueryPool(QueryPool& query_pool,
                      std::optional<uint32_t> first_query,
                      std::optional<uint32_t> query_count) override {
    maybe_throw();
  }
  void writeTimestamp(QueryPool& query_pool,
                      PipelineStageBits pipeline_stage,
                      uint32_t query_id) override {
    maybe_throw();
  }

  void beginQuery(QueryPool& query_pool) override { maybe_throw(); }
  void endQuery(QueryPool& query_pool) override { maybe_throw(); }

 private:
  const DeviceContext& device_ctx_;
};

// Check if a pointer is aligned
bool is_aligned(void* ptr, uint32_t alignment) {
  return reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
}

//
// Tests
//

//
// MemoryArena tests
//
// Test that suballocations are properly aligned
TEST(MemoryArenaTest, TestAlignment) {
  MemoryArena arena(256);
  constexpr uint32_t alloc_size = 2;
  for (uint32_t alignment = 1; alignment < 7; ++alignment) {
    auto* ptr = arena.alloc(alloc_size, 1u << alignment);
    EXPECT_EQ(is_aligned(ptr, 1u << alignment), true);
  }
}

//
// CommandList tests
//
TYPED_TEST(TypedDeviceContextTest, CommandListTest) {
  auto& device_context = *this->device_contexts_[0];

  CommandListBuilder cmd_builder(device_context);
  ExecutorMock executor(device_context);
  CommandList cmd_list(executor);
  cmd_builder.populateCommandList(cmd_list);

  EXPECT_EQ(cmd_list.getCount(), CommandListBuilder::cmd_count);
  EXPECT_EQ(cmd_list.hasCommands(), true);

  CommandListIterator list_itr(cmd_list);

  for (int32_t i = 0; i < CommandListBuilder::cmd_count; ++i) {
    ASSERT_EQ(list_itr.hasCommandsRemaining(), true) << "expected command at i=" << i;
    auto* cmd = list_itr.getNextCommand();
    ASSERT_NE(cmd, nullptr) << "null command i=" << i;
    if (cmd != nullptr) {
      cmd->execute(executor);
    }
  }
  EXPECT_EQ(list_itr.hasCommandsRemaining(), false);

  cmd_list.clear();
  EXPECT_EQ(cmd_list.getCount(), 0);
  EXPECT_EQ(cmd_list.hasCommands(), false);

  // Test exception
  cmd_builder.populateCommandList(cmd_list);
  executor.do_throw = true;
  EXPECT_THROW(cmd_list.flush("GfxCommandTest"), TestException);
  EXPECT_EQ(cmd_list.hasCommands(), false);
}

}  // namespace GfxDriverTests
