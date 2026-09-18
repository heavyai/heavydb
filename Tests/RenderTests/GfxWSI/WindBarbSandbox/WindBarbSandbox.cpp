/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Tests/RenderTests/GfxWSI/WindBarbSandbox/WindBarbSandbox.h"

#include <iostream>
#include <sstream>
#include <vector>

#include <imgui/imgui.h>
#include <glm/gtx/vector_angle.hpp>

#include "GfxDriver/Colors/GlmUtils.h"
#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/WSI/ImGuiBridge.h"
#include "GfxDriver/GfxContext.h"
#include "GfxDriver/Math/Matrix2d.h"
#include "GfxDriver/Pipeline/PipelineDescriptor.h"
#include "GfxDriver/Resources/BufferLayout.h"
#include "GfxDriver/Resources/ResourceManager.h"
#include "GfxDriver/Resources/VertexBuffer.h"
#include "GfxDriver/ShaderCompiler/GlslStructBuilder.h"
#include "GfxDriver/ShaderCompiler/ShaderManager.h"
#include "GfxDriver/Utils/NoiseUtils.h"
#include "GfxDriver/WSI/WindowSystemIntegration.h"
#include "Logger/Logger.h"
#include "QueryRenderer/Marks/Enums.h"
#include "QueryRenderer/Marks/WindBarbDefinitions.h"
#include "Shared/measure.h"
#include "Tests/RenderTests/Utils/AttachmentUtils.h"
#include "Tests/RenderTests/Utils/RenderPropertyUtils.h"

using namespace gfx;
using namespace QueryRenderer;

static constexpr uint32_t kMaxFieldDensity = 100;

static WindBarbDefinitions g_barb_defs = {};
static std::vector<std::string> g_shape_strings;

static std::string general_help_text =
    R"general_help(WindBarb rendering sandbox
Press 'u' to open the main UI and select the Help tab for interactive usage
)general_help";

void WindBarbSandbox::appPrintHelp(std::ostream& os) {
  os << general_help_text << "\n";
}

void WindBarbSandbox::addOrModifyProgramOptions(WSIAppOptions& options) {
  // Override default options and reinit
  options.window_width = 1800;
  options.window_height = 1200;
  options.num_samples = 4;
  options.setOptions();

  // TODO: add more options
}

void WindBarbSandbox::shaderLibraryInit(Library& library) {
  library.addFromManifestFile("ShaderManifest.json",
                              std::string(GFX_DRIVER_PATH) + "Render/shaders/");
  library.addFromManifestFile("ShaderManifest.json",
                              std::string(RENDER_TESTS_PATH) + "GfxWSI/shaders/");
  library.addFromManifestFile("ShaderManifest.json",
                              std::string(RENDER_TESTS_PATH) + "../../QueryRenderer/");
}

// Non-uniform ordering must match attr_pairs ordering in create_primitive_assembly
static std::vector<PropertyInfo> single_grid_properties = {
    {"x", false, BufferAttrType::kFloat},
    {"y", false, BufferAttrType::kFloat},
    {"size", false, BufferAttrType::kFloat},
    {"speed", false, BufferAttrType::kFloat},
    {"anchorScale", true, BufferAttrType::kFloat},
    {"quantizeDirection", true, BufferAttrType::kBool},
    {"opacity", true, BufferAttrType::kFloat},
    {"strokeWidth", true, BufferAttrType::kFloat},
    {"direction", true, BufferAttrType::kFloat},
    {"fillColor", true, BufferAttrType::kVec4f},
    {"strokeColor", true, BufferAttrType::kVec4f}};

// Non-uniform ordering must match attr_pairs ordering in create_primitive_assembly
static std::vector<PropertyInfo> field_properties = {
    {"x", false, BufferAttrType::kFloat},
    {"y", false, BufferAttrType::kFloat},
    {"size", false, BufferAttrType::kFloat},
    {"speed", false, BufferAttrType::kFloat},
    {"anchorScale", true, BufferAttrType::kFloat},
    {"quantizeDirection", true, BufferAttrType::kBool},
    {"opacity", true, BufferAttrType::kFloat},
    {"strokeWidth", true, BufferAttrType::kFloat},
    {"direction", false, BufferAttrType::kFloat},
    {"fillColor", false, BufferAttrType::kVec4f},
    {"strokeColor", false, BufferAttrType::kVec4f}};

void replace_common_markers(ShaderManager::BuilderUqPtrVector& builders,
                            uint32_t num_samples,
                            bool show_billboards) {
  // Vertex shader
  builders[0]->replaceFirstTag("numid", "0");
  builders[0]->replaceFirstTag("useKey", "0");

  std::string num_barb_types_str = std::to_string(g_barb_defs.pennant_counts.size());

  // Fragment shader
  builders[0]->replaceFirstTag("numBarbTypes", num_barb_types_str);
  builders[1]->replaceFirstTag("numBarbTypes", num_barb_types_str);
  builders[1]->replaceFirstTag("numBarbVerts", std::to_string(g_barb_defs.verts.size()));
  builders[1]->replaceFirstTag("showBillboard", std::to_string(show_billboards));
  builders[1]->replaceFirstTag("isMultiSampling", num_samples > 1 ? "1" : "0");

  builders[1]->appendTemplate("WSITests/windbarbSandbox_FragMain.glsl");

  // Geometry shader
  builders[2]->replaceFirstTag("numBarbTypes", num_barb_types_str);
  builders[2]->replaceFirstTag("doMirrorY", "1");
}

void WindBarbSandbox::initMaterialsAndPipelines() {
  auto& resource_mgr = device_->getResourceManager();
  auto const& shader_mgr = this->gfx_context_->getShaderManager();

  //
  // Generate spirv and materials
  //
  auto create_builders = [&, this]() {
    auto builders = shader_mgr.createBuilderVector({{"Marks/windBarbTemplate.vert"},
                                                    {"Marks/windBarbTemplate.frag"},
                                                    {"Marks/windBarbTemplate.geom"}});
    replace_common_markers(builders, num_samples_, show_billboards_);
    return builders;
  };

  // Single / grid material
  {
    auto builders = create_builders();
    auto [type_info_ss, vertex_attr_ss, prop_getter_ss, ubo_struct_builder] =
        create_render_prop_injectors(single_grid_properties, "WIND_BARB_VERT_UBO_TYPE");
    builders[0]->replaceFirstTag("RenderPropertyTypeInfos", type_info_ss.str());
    builders[0]->replaceFirstTag("VertexProperties", vertex_attr_ss.str());
    builders[0]->replaceFirstTag("UniformProperties",
                                 ubo_struct_builder.createStructString());
    builders[0]->replaceFirstTag("PropertyGetters", prop_getter_ss.str());
    builders[2]->replaceFirstTag("useUdirection", "1");

    // interface blocks
    gfx::GlslStructBuilder fragment_inputs("FragmentShaderInputs");
    gfx::GlslStructBuilder geometry_inputs("GeometryShaderInputs");
    generate_wind_barb_interface_blocks(geometry_inputs, fragment_inputs, true);

    builders[0]->replaceFirstTag("GeometryShaderInputs",
                                 geometry_inputs.createInterfaceBlockString(true));
    builders[2]->replaceFirstTag(
        "GeometryShaderInputs",
        geometry_inputs.createInterfaceBlockString(true, std::nullopt, true));

    auto fragment_inputs_str = fragment_inputs.createInterfaceBlockString(true);
    builders[1]->replaceFirstTag("FragmentShaderInputs", fragment_inputs_str);
    builders[2]->replaceFirstTag("FragmentShaderInputs", fragment_inputs_str);

    auto spirv_caches = shader_mgr.createCacheVector(std::move(builders), true);

    material_ = resource_mgr.createMaterial("Barbs", spirv_caches);
    material_->updateDescriptorSets();
    updateBarbDefinitionUniforms(*material_);
  }
  // Field material
  {
    auto builders = create_builders();
    auto [type_info_ss, vertex_attr_ss, prop_getter_ss, ubo_struct_builder] =
        create_render_prop_injectors(field_properties, "WIND_BARB_VERT_UBO_TYPE");
    builders[0]->replaceFirstTag("RenderPropertyTypeInfos", type_info_ss.str());
    builders[0]->replaceFirstTag("VertexProperties", vertex_attr_ss.str());
    builders[0]->replaceFirstTag("UniformProperties",
                                 ubo_struct_builder.createStructString());
    builders[0]->replaceFirstTag("PropertyGetters", prop_getter_ss.str());

    builders[2]->replaceFirstTag("useUdirection", "0");

    gfx::GlslStructBuilder fragment_inputs("FragmentShaderInputs");
    gfx::GlslStructBuilder geometry_inputs("GeometryShaderInputs");
    generate_wind_barb_interface_blocks(geometry_inputs, fragment_inputs, false);

    builders[0]->replaceFirstTag("GeometryShaderInputs",
                                 geometry_inputs.createInterfaceBlockString(true));
    builders[2]->replaceFirstTag(
        "GeometryShaderInputs",
        geometry_inputs.createInterfaceBlockString(true, std::nullopt, true));

    auto fragment_inputs_str = fragment_inputs.createInterfaceBlockString(true);
    builders[1]->replaceFirstTag("FragmentShaderInputs", fragment_inputs_str);
    builders[2]->replaceFirstTag("FragmentShaderInputs", fragment_inputs_str);

    auto spirv_caches = shader_mgr.createCacheVector(std::move(builders), true);

    field_material_ = resource_mgr.createMaterial("Field Barbs", spirv_caches);
    field_material_->updateDescriptorSets();
    updateBarbDefinitionUniforms(*field_material_);
  }

  //
  // Create PrimitiveAssemblies
  //
  auto create_primitive_assembly =
      [&](DistributionData& data, Material& material, bool is_field) {
        VboAttrToShaderAttrPairs attr_pairs = {
            {"x", "x"}, {"y", "y"}, {"size", "size"}, {"speed", "speed"}};

        if (is_field) {
          attr_pairs.push_back({"direction", "direction"});
          attr_pairs.push_back({"fillColor", "fillColor"});
          attr_pairs.push_back({"strokeColor", "strokeColor"});
        }

        PrimitiveAssemblyAttrInfo attr_info = {
            {static_cast<VertexBuffer*>(data.vbo.get()), data.buffer_layout}, attr_pairs};

        data.primitive_assembly = resource_mgr.createPrimitiveAssembly(
            data.name, PrimitiveTopology::kPointList, material, attr_info);
      };

  create_primitive_assembly(distribution_data_[Distribution::kSingle], *material_, false);
  create_primitive_assembly(distribution_data_[Distribution::kGrid], *material_, false);
  create_primitive_assembly(
      distribution_data_[Distribution::kField], *field_material_, true);

  //
  // Create pipelines
  //

  PipelineDescriptor pipeline_desc;
  pipeline_desc.setRasterSampleCount(raster_sample_count_);

  // Single / grid mode pipeline
  pipeline_ = resource_mgr.createGraphicsPipeline(
      "Barbs",
      *material_,
      pipeline_desc,
      distribution_data_[Distribution::kGrid].primitive_assembly.get());
  pipeline_->create(*render_pass_);

  // Field mode pipeline
  field_pipeline_ = resource_mgr.createGraphicsPipeline(
      "Field Barbs",
      *field_material_,
      pipeline_desc,
      distribution_data_[Distribution::kField].primitive_assembly.get());
  field_pipeline_->create(*render_pass_);
}

void WindBarbSandbox::appInit() {
  QueryRenderer::init_wind_barb_defs(g_barb_defs);

  for (int type = 0; type < (int)SymbolShapeType::kCOUNT; ++type) {
    g_shape_strings.push_back(to_string((SymbolShapeType)type));
  }

  auto& resource_mgr = device_->getResourceManager();

  //
  // Create render targets, framebuffer, and renderpasses
  //

  // attachment manager and textures
  auto color_format = wsi_->getWindowPixelFormat();
  render_targets_ = build_attachments(resource_mgr,
                                      window_width_,
                                      window_height_,
                                      {{color_format, Framebuffer::Attachment::kColor0}},
                                      "Framebuffer",
                                      num_samples_);

  // renderpass and framebuffer
  render_pass_clear_ =
      resource_mgr.createRenderPass("Clearing",
                                    render_targets_.attachment_mgr.getLayout(),
                                    gfx::RenderPass::ClearBits::kAll,
                                    ImageLayout::kUndefined,
                                    ImageLayout::kAttachment);

  render_pass_ = resource_mgr.createRenderPass("No clearing",
                                               render_targets_.attachment_mgr.getLayout(),
                                               gfx::RenderPass::ClearBits::kNone,
                                               ImageLayout::kAttachment,
                                               ImageLayout::kAttachment);

  framebuffer_ = resource_mgr.createFramebuffer("Framebuffer",
                                                *render_pass_,
                                                render_targets_.attachment_mgr,
                                                window_width_,
                                                window_height_,
                                                num_samples_);

  initVertexBuffers();
  initMaterialsAndPipelines();
  static std::vector<Background::Type> bg_types{Background::Type::kNone,
                                                Background::Type::kSolidColor,
                                                Background::Type::kGridLines};
  background_ = std::make_unique<Background>(*device_, bg_types);
  background_->init(raster_sample_count_, *render_pass_);
  background_->setType(Background::Type::kSolidColor);

  //
  // Init UI
  //
  imgui_bridge_->initBackendAndResources(
      framebuffer_->getAttachmentManager().getLayout());

  wsi_->registerEventHandler([this](const WSIEvent& event) { handleWSIEvent(event); });
}

// BarbVertex struct
// Used to populate vertex buffers
// Must match buffer_layout_
struct BarbVertex {
  float x;
  float y;
  float size;
  float speed;
};

// FieldBarbVertex struct
// Used to populate vertex buffers
// Must match field_buffer_layout_
struct FieldBarbVertex {
  float x;
  float y;
  float size;
  float speed;
  float direction;
  glm::vec4 fill_color;
  glm::vec4 stroke_color;
};

template <typename T>
void create_distribution_buffer(ResourceManager& resource_mgr,
                                DistributionData& data,
                                const std::vector<T>& vertices,
                                const std::string& name,
                                bool is_field) {
  data.name = name;
  data.num_barbs = vertices.size();
  uint64_t vertex_size{0};

  data.buffer_layout = std::make_shared<InterleavedBufferLayout>();
  CHECK(data.buffer_layout);
  data.buffer_layout->addAttribute("x", BufferAttrType::kFloat);
  data.buffer_layout->addAttribute("y", BufferAttrType::kFloat);
  data.buffer_layout->addAttribute("size", BufferAttrType::kFloat);
  data.buffer_layout->addAttribute("speed", BufferAttrType::kFloat);
  if (is_field) {
    data.buffer_layout->addAttribute("direction", BufferAttrType::kFloat);
    data.buffer_layout->addAttribute("fillColor", BufferAttrType::kVec4f);
    data.buffer_layout->addAttribute("strokeColor", BufferAttrType::kVec4f);
  }
  vertex_size = sizeof(T);

  auto buffer_size = data.num_barbs * vertex_size;
  data.vbo = resource_mgr.createBuffer(
      name + " VBO",
      {BufferType::kVertexBuffer, buffer_size, BufferUsageBits::kLayoutBufferBit});

  data.vbo->updateSubDataWithLayout(vertices.data(), buffer_size, 0, data.buffer_layout);
}

void WindBarbSandbox::initVertexBuffers() {
  //
  // Populate DistributionData and create vertex buffer
  //
  updateSingleDistribution();
  updateGridDistribution();
  updateFieldDistribution();
}

void WindBarbSandbox::updateBarbDefinitionUniforms(Material& material) {
  material.setUniformAttribute("uBarbCounts", g_barb_defs.barb_counts);
  material.setUniformAttribute("uBarbOffsets", g_barb_defs.barb_offsets);
  material.setUniformAttribute("uPennantCounts", g_barb_defs.pennant_counts);
  material.setUniformAttribute("uPennantOffsets", g_barb_defs.pennant_offsets);
  material.setUniformAttribute("uBarbVerts", g_barb_defs.verts);
}

void WindBarbSandbox::updateSingleDistribution() {
  auto& data = distribution_data_[Distribution::kSingle];
  data.num_barbs = 1;
  float size = static_cast<float>(std::min(window_width_, window_height_)) * 0.5f;

  BarbVertex v = {};
  v.x = size;
  v.y = size;
  v.size = size;
  v.speed = static_cast<float>(single_barb_speed_);

  if (data.vbo) {
    data.vbo->updateSubData(&v, sizeof(BarbVertex), 0);
  } else {
    create_distribution_buffer<BarbVertex>(
        device_->getResourceManager(), data, {v}, "Single", false);
  }
  single_distribution_dirty_ = false;
}

void WindBarbSandbox::updateGridDistribution() {
  static constexpr uint32_t kGridColumns = 6;
  static constexpr uint32_t kGridRows = 5;

  auto& data = distribution_data_[Distribution::kGrid];
  data.num_barbs = kGridColumns * kGridRows;

  std::vector<BarbVertex> vertex_data(data.num_barbs);

  float fwindow_width = static_cast<float>(window_width_);
  float fwindow_height = static_cast<float>(window_height_);
  float size = fwindow_width / static_cast<float>(kGridColumns) * 0.5f;

  float fx_step = 1.0f / kGridColumns;
  float fy_step = 1.0f / kGridRows;
  float fy = fy_step * 0.75f;
  float speed = 0.0f;
  uint32_t barb_id = 0;
  for (uint32_t y = 0; y < kGridRows; y++, fy += fy_step) {
    float fx = fx_step * 0.3f;
    for (uint32_t x = 0; x < kGridColumns && barb_id < 30;
         x++, fx += fx_step, speed += 5.0f, barb_id++) {
      auto& vertex = vertex_data[barb_id];
      vertex.x = fx * fwindow_width;
      vertex.y = fy * fwindow_height;
      vertex.size = size;
      vertex.speed = speed;
    }
  }

  if (data.vbo) {
    data.vbo->updateSubData(
        vertex_data.data(), vertex_data.size() * sizeof(BarbVertex), 0);
  } else {
    create_distribution_buffer<BarbVertex>(
        device_->getResourceManager(), data, vertex_data, "Grid", false);
  }

  grid_distribution_dirty_ = false;
}

void WindBarbSandbox::updateFieldDistribution() {
  auto& data = distribution_data_[Distribution::kField];

  data.num_barbs = field_density_ * field_density_;

  std::vector<FieldBarbVertex> vertex_data(data.num_barbs);
  field_draw_count_ = data.num_barbs;

  float fwindow_width = static_cast<float>(window_width_);
  float fwindow_height = static_cast<float>(window_height_);

  float noise_scale = 1.0f / field_noise_size_;
  auto eval_noise = [this, &noise_scale](float x, float y) -> float {
    float n = fractal_noise(3,
                            x - 0.5f + field_noise_center_.x,
                            y - 0.5f + field_noise_center_.y,
                            noise_scale);
    n = n * 1.2f - 0.1f;  // black / white point clipping so we see peaks
    n = n * (field_max_speed_ - field_min_speed_) + field_min_speed_;
    std::clamp(n, field_min_speed_, field_max_speed_);
    return n;
  };

  float fstep = 1.0f / field_density_;
  float fy = fstep * 0.75f;
  uint32_t vertex_id = 0;
  for (int y = 0; y < field_density_; y++, fy += fstep) {
    float fx = fstep * 0.3f;
    for (int x = 0; x < field_density_; x++, fx += fstep, vertex_id++) {
      auto& vertex = vertex_data[vertex_id];
      vertex.x = fx * fwindow_width;
      vertex.y = fy * fwindow_height;
      vertex.size = field_barb_size_;

      // compute direction along noise gradient
      float np = eval_noise(fx, fy);
      float nx = eval_noise(fx + 0.0001f, fy);
      float ny = eval_noise(fx, fy + 0.0001f);
      vertex.speed = np;
      auto dir = glm::normalize(glm::vec2(np - nx, np - ny));
      vertex.direction = glm::degrees(std::atan2(dir.y, dir.x));
      if (field_use_auto_color_) {
        auto color = HLSAtoRGBA(vertex.speed + 230.f, 0.75f, 1.0f, 1.0f);
        vertex.fill_color = color;
        vertex.stroke_color = color;
      } else {
        vertex.fill_color = fill_color_;
        vertex.stroke_color = stroke_color_;
      }
    }
  }

  auto required_size = vertex_data.size() * sizeof(FieldBarbVertex);
  if (data.vbo && data.vbo->getNumBytes() < required_size) {
    device_->getResourceManager().destroyBuffer(std::move(data.vbo));
  }

  if (data.vbo) {
    if (data.vbo->getNumBytes() < required_size) {
      data.vbo->rebuild(vertex_data.data(), required_size);
    } else {
      data.vbo->updateSubData(vertex_data.data(), required_size, 0);
    }
  } else {
    create_distribution_buffer<FieldBarbVertex>(
        device_->getResourceManager(), data, vertex_data, "Field", true);
  }

  field_distribution_dirty_ = false;
}

void WindBarbSandbox::maybeResetTime() {
  // Reset timing stats if anything changed that would invalidate current totals
  if ((distribution_ == Distribution::kSingle && single_distribution_dirty_) ||
      (distribution_ == Distribution::kGrid && grid_distribution_dirty_) ||
      (distribution_ == Distribution::kField && field_distribution_dirty_) ||
      uniforms_dirty_ || view_uniforms_dirty_ || render_time_dirty_) {
    num_renders_ = 0;
    total_time_ms_ = 0.0;
    render_time_dirty_ = false;
  }
}

void WindBarbSandbox::updateUniforms() {
  auto update_uniforms = [this](Material& material, bool is_field) {
    if (uniforms_dirty_) {
      material.setUniformAttribute("opacity", opacity_);
      material.setUniformAttribute("anchorScale", anchor_scale_);
      material.setUniformAttribute("uInvViewportWidth", 1.0f / (float)window_width_);
      material.setUniformAttribute("uInvViewportHeight", 1.0f / (float)window_height_);
      material.setUniformAttribute("quantizeDirection", do_quantize_direction_);
      material.setUniformAttribute("strokeWidth", stroke_width_);

      // Direction and colors are vertex properties in Field distribution
      if (!is_field) {
        material.setUniformAttribute("direction", direction_);
        float direction =
            do_quantize_direction_ ? (((int)direction_ + 5) / 10) * 10.0f : direction_;
        direction = direction * 3.14159265359f / 180.0f;
        material.setUniformAttribute("uSinDirection", std::sin(direction));
        material.setUniformAttribute("uCosDirection", std::cos(direction));

        material.setUniformAttribute("fillColor", fill_color_);
        material.setUniformAttribute("strokeColor", stroke_color_);
      }
    }
  };

  update_uniforms(*material_, false);
  update_uniforms(*field_material_, true);

  if (view_uniforms_dirty_) {
    auto fwidth = static_cast<float>(window_width_);
    auto fheight = static_cast<float>(window_height_);
    auto proja = 2.0f / fwidth;
    auto projb = 2.0f / fheight;
    gfx::Math::Matrix2d<float> tmp = {{{proja, 0, 0, projb, -1, -1}}};

    material_->setUniformAttribute("uVPmatrix", tmp.getDataArrayRef());
    field_material_->setUniformAttribute("uVPmatrix", tmp.getDataArrayRef());
  }
  uniforms_dirty_ = false;
  view_uniforms_dirty_ = false;
}

void WindBarbSandbox::render() {
  maybeResetTime();

  // Update distribution vertex buffer
  if (distribution_ == Distribution::kSingle && single_distribution_dirty_) {
    updateSingleDistribution();
  }
  if (distribution_ == Distribution::kGrid && grid_distribution_dirty_) {
    updateGridDistribution();
  }
  if (distribution_ == Distribution::kField && field_distribution_dirty_) {
    updateFieldDistribution();
  }

  updateUniforms();

  // Draw points
  auto& cmd_list = device_->getCommandList();

  // The first draw needs to clear the framebuffer
  RenderPass* render_pass = render_pass_clear_.get();

  // Draw the background if enabled
  if (background_->getType() != Background::Type::kNone) {
    cmd_list.beginRenderPass(*render_pass, *framebuffer_);
    background_->draw(cmd_list);
    cmd_list.endRenderPass().flush("Background");

    // Switch to non-clearing RenderPass
    render_pass = render_pass_.get();
  }

  // Draw the barbs
  auto clock_start = timer_start();
  auto& data = distribution_data_[distribution_];

  cmd_list.beginRenderPass(*render_pass, *framebuffer_);
  if (distribution_ == Distribution::kField) {
    cmd_list.drawVertices(*field_pipeline_,
                          *static_cast<VertexBuffer*>(data.vbo.get()),
                          static_cast<uint32_t>(field_draw_count_));
  } else {
    cmd_list.drawVertices(
        *pipeline_, *static_cast<VertexBuffer*>(data.vbo.get()), data.num_barbs);
  }
  cmd_list.endRenderPass().flush("Draw");

  // Add the time to the running total
  auto time =
      timer_stop<std::chrono::steady_clock::time_point, std::chrono::microseconds>(
          clock_start);
  total_time_ms_ += static_cast<double>(time) / 1000.0;
  num_renders_++;

  // Draw all UI elements
  drawUI();

  // Copy color to swapchain and present
  wsi_->copyAndPresentTexture(*render_targets_.textures[0]);
}

void WindBarbSandbox::drawUI() {
  imgui_bridge_->beginRecording();

  //
  // Main panel
  //
  if (show_ui_) {
    imgui_bridge_->setNextWindowPos(ImGuiBridge::kTopRight, 5.0f, ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(640, 1120), ImGuiCond_Once);
    if (ImGui::Begin("Wind Barbs", &show_ui_)) {
      if (ImGui::BeginTabBar("Tabs", ImGuiTabBarFlags_None)) {
        if (ImGui::BeginTabItem("Settings")) {
          //
          // Distribution
          //
          if (ImGui::CollapsingHeader("Distribution", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::RadioButton("Single", distribution_ == Distribution::kSingle)) {
              distribution_ = Distribution::kSingle;
              render_time_dirty_ = true;
            }
            ImGui::SameLine();

            {
              ImGuiBridge::ScopedDisable guard(distribution_ == Distribution::kSingle);
              if (ImGui::SliderInt(
                      "##Speed", (int*)&single_barb_speed_, 0, (int)kMaxWindBarbSpeed)) {
                single_distribution_dirty_ = true;
              }
            }

            ImGui::Separator();

            // Grid mode
            if (ImGui::RadioButton("Grid", distribution_ == Distribution::kGrid)) {
              distribution_ = Distribution::kGrid;
              render_time_dirty_ = true;
            }

            ImGui::Separator();

            // Random mode
            if (ImGui::RadioButton("Field", distribution_ == Distribution::kField)) {
              distribution_ = Distribution::kField;
              render_time_dirty_ = true;
            }

            {  // Random distribution controls
              ImGuiBridge::ScopedDisable guard(distribution_ == Distribution::kField);
              field_distribution_dirty_ |= ImGui::SliderFloat(
                  "Barb Size", &field_barb_size_, 1.0f, 200.0f, "%.2f");
              ImGui::Spacing();
              if (ImGui::SliderInt(
                      "Barb Density", &field_density_, 1, kMaxFieldDensity)) {
                field_distribution_dirty_ = true;
              }
              if (field_density_ < 1) {
                field_density_ = 1;
              }
              ImGui::Spacing();
              if (ImGui::SliderFloat("Noise Size", &field_noise_size_, 0.01f, 2.0f)) {
                field_distribution_dirty_ = true;
              }
              ImGui::Spacing();
              field_distribution_dirty_ |=
                  ImGui::DragFloatRange2("Wind Speed",
                                         &field_min_speed_,
                                         &field_max_speed_,
                                         5.0f,
                                         0.0f,
                                         kMaxWindBarbSpeed,
                                         "Min: %.2f",
                                         "Max: %.2f",
                                         ImGuiSliderFlags_AlwaysClamp);
              imgui_bridge_->HelpMarker(
                  "left-click and drag left/right to change value\n"
                  "ctrl + left-click to type value");
              ImGui::Spacing();
              if (ImGui::Checkbox("Use gradient color", &field_use_auto_color_)) {
                field_distribution_dirty_ = true;
              }
            }
          }

          //
          // Properties
          //
          if (ImGui::CollapsingHeader("Common Properties",
                                      ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Checkbox("Show Billboards", &show_billboards_)) {
              reloadShaders();
            }
            ImGui::Separator();
            {
              ImGuiBridge::ScopedDisable guard(distribution_ != Distribution::kField);
              uniforms_dirty_ |=
                  ImGui::SliderFloat("Direction", &direction_, 0.0f, 360.0f);
            }
            uniforms_dirty_ |=
                ImGui::Checkbox("Quantize Direction", &do_quantize_direction_);

            ImGui::Separator();

            uniforms_dirty_ |= ImGui::SliderFloat("Opacity", &opacity_, 0.0f, 1.0f);
            uniforms_dirty_ |=
                ImGui::SliderFloat("Anchor Scale", &anchor_scale_, 0.0f, 1.0f);

            {
              ImGuiBridge::ScopedDisable guard(!field_use_auto_color_ ||
                                               (distribution_ != kField));
              static constexpr ImGuiColorEditFlags color_edit_flags =
                  ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel |
                  ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_AlphaBar;
              bool colors_dirty =
                  ImGui::ColorEdit4("Fill Color", (float*)&fill_color_, color_edit_flags);
              ImGui::Separator();
              colors_dirty |= ImGui::ColorEdit4(
                  "Stroke Color", (float*)&stroke_color_, color_edit_flags);
              if (colors_dirty) {
                if (distribution_ == Distribution::kField) {
                  field_distribution_dirty_ = true;
                }
                // Ensure uniforms for the single/grid material get updated too
                uniforms_dirty_ = true;
              }
            }
            uniforms_dirty_ |=
                ImGui::SliderFloat("Stroke Width", &stroke_width_, 0.0f, 50.0f);
          }
          ImGui::EndTabItem();
        }

        // Background
        if (ImGui::BeginTabItem("Background")) {
          background_->drawUI();
          ImGui::EndTabItem();
        }

        // Memory Summary
        if (ImGui::BeginTabItem("Memory")) {
          std::stringstream ss;
          device_->getResourceManager().logMemorySummary(ss);
          imgui_bridge_->pushFont(ImGuiBridge::Font::kFixedWidth);
          ImGui::Text("%s", ss.str().c_str());
          imgui_bridge_->popFont();
          ImGui::EndTabItem();
        }

        // HELP!!!!
        static HelpInputTable hotkeys = {
            {{"", "D", "Cycle distribution"},
             {"", "S", "Single-barb mode increment speed"},
             {"", "G", "Gradient color by speed (field)"},
             {"", "B", "Cycle background mode"}},
            {{"", "R", "Reload shaders"}},
            {{"", "T", "Show time overlay"}, {"", "U", "Show UI"}}};

        static HelpInputTableGroup mouse_actions = {
            {"", "L drag", "Pan Field Noise"}, {"", "V scroll", "Scale Field Noise"}};

        if (ImGui::BeginTabItem("Help")) {
          ImGui::CollapsingHeader("Hotkeys", ImGuiTreeNodeFlags_Leaf);
          draw_hotkey_help(hotkeys, true, false);

          ImGui::CollapsingHeader("Mouse / Trackpad", ImGuiTreeNodeFlags_Leaf);
          draw_mouse_action_help_table(mouse_actions, true, false);

          ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
      }
    }
    ImGui::End();
  }

  //
  // Time overlay
  //
  static ImGuiBridge::Overlay time_overlay(
      *imgui_bridge_,
      "Time",
      ImGuiBridge::Font::kFixedWidth,
      ImGuiBridge::WindowPos::kBottomRight,
      show_time_overlay_,
      [this]() {
        ImGui::Text("Avg Time (ms): %.3f",
                    total_time_ms_ / static_cast<double>(num_renders_));
      });

  if (show_time_overlay_) {
    time_overlay.draw();
  }

  imgui_bridge_->endRecordingAndDraw(*framebuffer_);
}

void WindBarbSandbox::destroyMaterialsAndPipelines() {
  auto& resource_mgr = device_->getResourceManager();
  if (pipeline_) {
    resource_mgr.destroyPipeline(std::move(pipeline_));
  }
  if (field_pipeline_) {
    resource_mgr.destroyPipeline(std::move(field_pipeline_));
  }
  for (auto& data : distribution_data_) {
    data.primitive_assembly = nullptr;
  }

  material_ = nullptr;
  field_pipeline_ = nullptr;
}

void WindBarbSandbox::appShutdown() {
  auto& resource_mgr = device_->getResourceManager();
  destroyMaterialsAndPipelines();
  background_->shutdown();
  if (render_pass_clear_) {
    resource_mgr.destroyRenderPass(std::move(render_pass_clear_));
  }
  if (render_pass_) {
    resource_mgr.destroyRenderPass(std::move(render_pass_));
  }
  for (auto& data : distribution_data_) {
    data.primitive_assembly = nullptr;
    if (data.vbo) {
      resource_mgr.destroyBuffer(std::move(data.vbo));
    }
  }
}

void WindBarbSandbox::reloadShaders() {
  LOG(INFO) << "Reloading shaders";
  destroyMaterialsAndPipelines();
  background_->shutdown();

  auto library = std::make_unique<Library>();
  shaderLibraryInit(*library);
  const_cast<ShaderManager&>(gfx_context_->getShaderManager())
      .replaceLibrary(std::move(library));

  initMaterialsAndPipelines();
  background_->init(raster_sample_count_, *render_pass_);
  uniforms_dirty_ = true;
  view_uniforms_dirty_ = true;
}

void WindBarbSandbox::handleWSIEvent(const WSIEvent& event) {
  using e = WSIEvent::Type;
  switch (event.type()) {
    case e::kWindowResize:
      handleWindowResizeEvent(static_cast<const WSIWindowResizeEvent&>(event));
      break;
    case e::kMouseButton:
      handleMouseButtonEvent(static_cast<const WSIMouseButtonEvent&>(event));
      break;
    case e::kCursor:
      handleCursorEvent(static_cast<const WSIMouseCursorEvent&>(event));
      break;
    case e::kScroll:
      handleScrollEvent(static_cast<const WSIScrollEvent&>(event));
      break;
    case e::kKeyboard:
      handleKeyboardEvent(static_cast<const WSIKeyboardEvent&>(event));
      break;
  }
}

void WindBarbSandbox::handleWindowResizeEvent(const WSIWindowResizeEvent& event) {
  single_distribution_dirty_ = true;
  grid_distribution_dirty_ = true;
  field_distribution_dirty_ = true;
  uniforms_dirty_ = true;
  view_uniforms_dirty_ = true;
  background_->setImageSize(event.width(), event.height());
}

void WindBarbSandbox::handleMouseButtonEvent(const WSIMouseButtonEvent& event) {
  auto& io = ImGui::GetIO();
  auto button = event.button();
  auto action = event.action();
  if ((!io.WantCaptureMouse) && button == WSIMouseButton::kLeft &&
      action == WSIMouseAction::kPress) {
    is_capturing_mouse_ = true;
  } else if (button == WSIMouseButton::kLeft && action == WSIMouseAction::kRelease) {
    is_capturing_mouse_ = false;
  }
}

void WindBarbSandbox::handleCursorEvent(const WSIMouseCursorEvent& event) {
  auto x = event.x();
  auto y = event.y();
  if (is_capturing_mouse_) {
    glm::vec2 dcursor = (last_cursor_ - glm::vec2(x, y));
    field_noise_center_.x += dcursor.x / window_width_;
    field_noise_center_.y += dcursor.y / window_height_;
    field_distribution_dirty_ = true;
  }
  last_cursor_ = glm::vec2(x, y);
}

void WindBarbSandbox::handleScrollEvent(const WSIScrollEvent& event) {
  auto& io = ImGui::GetIO();
  if (!io.WantCaptureMouse) {
    field_noise_size_ += (event.y() * 0.1f);
    field_distribution_dirty_ = true;
  }
}

void WindBarbSandbox::handleKeyboardEvent(const WSIKeyboardEvent& event) {
  // Handle keys we care about first
  if (event.action() == WSIKeyboardAction::kPress) {
    switch (event.key()) {
      // Cycle background
      case WSIKeyboardKey::kB:
        background_->cycleType();
        break;

      // Cycle distribution
      case WSIKeyboardKey::kD: {
        int type = static_cast<int>(distribution_) + 1;
        if (type >= kNumDistributions) {
          type = 0;
        }
        distribution_ = static_cast<Distribution>(type);
      } break;

      // Toggle color by gradient by speed
      case WSIKeyboardKey::kG:
        field_use_auto_color_ = !field_use_auto_color_;
        field_distribution_dirty_ = true;
        break;

      // Cycle single symbol type
      case WSIKeyboardKey::kS:
        single_barb_speed_ += 5;
        if (single_barb_speed_ > kMaxWindBarbSpeed) {
          single_barb_speed_ = 0;
        }
        single_distribution_dirty_ = true;
        break;

      // Reload shaders
      case WSIKeyboardKey::kR:
        reloadShaders();
        break;

      // Toggle time overlay
      case WSIKeyboardKey::kT:
        show_time_overlay_ = !show_time_overlay_;
        break;

      // Toggle UI
      case WSIKeyboardKey::kU:
        show_ui_ = !show_ui_;
        break;

      default:
        break;
    }
  }
}

DECLARE_WSI_APP_MAIN(WindBarbSandbox)
