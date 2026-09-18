/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Tests/RenderTests/GfxWSI/SymbolSandbox/SymbolSandbox.h"

#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include <imgui/imgui.h>

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
#include "GfxDriver/WSI/WindowSystemIntegration.h"
#include "Logger/Logger.h"
#include "QueryRenderer/Marks/SymbolDefinitions.h"
#include "Shared/measure.h"
#include "Tests/RenderTests/Utils/AttachmentUtils.h"
#include "Tests/RenderTests/Utils/RenderPropertyUtils.h"

using namespace gfx;
using namespace QueryRenderer;

static constexpr uint32_t kMaxRandomSymbols = 1000000;
static constexpr uint32_t kGridColumns = 5;
static constexpr uint32_t kGridRows = 3;

static SymbolDefinitions g_symbol_defs = {};
static std::vector<std::string> g_shape_strings;

static std::string general_help_text =
    R"general_help(SymbolSandbox procedural symbol rendering sandbox
Press 'u' to open the main UI and select the Help tab for interactive usage
)general_help";

static std::string editor_help_text = R"editor_help([Set Edit Mode]
  Set Background to 'Editor'
  Set Distribution to 'Single'

[Reset Symbols]
  Restores all Symbol definitions to default

Table editing:
  ctrl+left click on slider to type value
)editor_help";

void SymbolSandbox::appPrintHelp(std::ostream& os) {
  os << general_help_text << "\n";
}

void SymbolSandbox::addOrModifyProgramOptions(WSIAppOptions& options) {
  // Override default options and reinit
  options.window_width = 1800;
  options.window_height = 1200;
  options.num_samples = 4;
  options.setOptions();

  // TODO: add more options
}

void SymbolSandbox::shaderLibraryInit(Library& library) {
  library.addFromManifestFile("ShaderManifest.json",
                              std::string(GFX_DRIVER_PATH) + "Render/shaders/");
  library.addFromManifestFile("ShaderManifest.json",
                              std::string(RENDER_TESTS_PATH) + "GfxWSI/shaders/");
  library.addFromManifestFile("ShaderManifest.json",
                              std::string(RENDER_TESTS_PATH) + "../../QueryRenderer/");
}

static std::vector<PropertyInfo> point_render_properties = {
    {"shape", false, BufferAttrType::kUint},
    {"x", false, BufferAttrType::kFloat},
    {"x2", true, BufferAttrType::kFloat},
    {"xc", true, BufferAttrType::kFloat},
    {"y", false, BufferAttrType::kFloat},
    {"y2", true, BufferAttrType::kFloat},
    {"yc", true, BufferAttrType::kFloat},

    {"width", false, BufferAttrType::kFloat},
    {"height", false, BufferAttrType::kFloat},

    {"opacity", true, BufferAttrType::kFloat},

    {"fillColor", true, BufferAttrType::kVec4f},
    {"fillOpacity", true, BufferAttrType::kFloat},

    {"strokeColor", true, BufferAttrType::kVec4f},
    {"strokeOpacity", true, BufferAttrType::kFloat},
    {"strokeWidth", true, BufferAttrType::kFloat},
    // Point specific properties
    {"uVPmatrix", true, BufferAttrType::kMat3x2f},
    {"uPivotx", true, BufferAttrType::kFloat},
    {"uPivoty", true, BufferAttrType::kFloat}};

static std::vector<PropertyInfo> geometry_render_properties = {
    {"shape", false, BufferAttrType::kUint},
    {"x", false, BufferAttrType::kFloat},
    {"x2", true, BufferAttrType::kFloat},
    {"xc", true, BufferAttrType::kFloat},
    {"y", false, BufferAttrType::kFloat},
    {"y2", true, BufferAttrType::kFloat},
    {"yc", true, BufferAttrType::kFloat},

    {"width", false, BufferAttrType::kFloat},
    {"height", false, BufferAttrType::kFloat},

    {"opacity", true, BufferAttrType::kFloat},

    {"fillColor", true, BufferAttrType::kVec4f},
    {"fillOpacity", true, BufferAttrType::kFloat},

    {"strokeColor", true, BufferAttrType::kVec4f},
    {"strokeOpacity", true, BufferAttrType::kFloat},
    {"strokeWidth", true, BufferAttrType::kFloat},
    // Geometry specific properties
    {"angle", true, gfx::BufferAttrType::kFloat},
    {"angleUnit", true, gfx::BufferAttrType::kUint}};

void inject_into_vertex_shader(ShaderManager::Builder& builder,
                               std::stringstream& type_info_stream,
                               const std::string& vbo_props_string,
                               const std::string& prop_getter_string,
                               const std::string& ubo_props_string) {
  builder.replaceFirstTag("VertexProperties", vbo_props_string);
  builder.replaceFirstTag("UniformProperties", ubo_props_string);
  builder.replaceFirstTag("RenderPropertyTypeInfos", type_info_stream.str());
  builder.replaceFirstTag("PropertyGetters", prop_getter_string);

  builder.replaceFirstTag("numid", "0");
  builder.replaceFirstTag("useKey", "0");

  builder.replaceFirstTag("doHeatmapEdgePad", "0");
  builder.replaceFirstTag("computeX", "0");
  builder.replaceFirstTag("computeY", "0");
  builder.replaceFirstTag("computeWidth", "0");
  builder.replaceFirstTag("computeHeight", "0");
}

void inject_into_fragment_shader(ShaderManager::Builder& builder,
                                 uint32_t num_samples,
                                 bool using_geometry_shader) {
  builder.replaceFirstTag("numSymbolVerts", std::to_string(g_symbol_defs.verts.size()));
  builder.replaceFirstTag("isMultiSampling", num_samples > 1 ? "1" : "0");
  builder.replaceFirstTag("usingGeomOrMeshShader", using_geometry_shader ? "1" : "0");

  builder.addSubroutineBinding(
      "mapDistanceToColor", "mapDistanceToColorFillAndStroke", true);

  builder.appendTemplate("WSITests/symbolSandbox_FragMain.glsl");
}

void inject_into_geometry_shader(ShaderManager::Builder& builder) {
  builder.replaceFirstTag("useUangle", "1");
  builder.replaceFirstTag("doAccumIndex", "0");
  builder.replaceFirstTag("doHeatmapEdgePad", "0");
}

void SymbolSandbox::initMaterialsAndPipelines() {
  auto& resource_mgr = device_->getResourceManager();
  auto const& shader_mgr = this->gfx_context_->getShaderManager();

  //
  // Generate spirv
  //

  // Point shader billboards
  auto builders = shader_mgr.createBuilderVector(
      {{"Marks/fastSymbolTemplate.vert"}, {"Marks/fastSymbolTemplate.frag"}});

  {
    // Create stringstreams and struct builder for render property injection
    auto [type_info_ss, vertex_attr_ss, prop_getter_ss, ubo_struct_builder] =
        create_render_prop_injectors(point_render_properties,
                                     "FAST_SYMBOL_VERT_UBO_TYPE");

    auto vertex_attr_str = vertex_attr_ss.str();
    auto prop_getter_str = prop_getter_ss.str();
    gfx::GlslStructBuilder fragment_inputs("FragmentShaderInputs");
    std::optional<gfx::GlslStructBuilder> geometry_inputs = std::nullopt;
    generate_fast_symbol_interface_blocks(
        fragment_inputs, geometry_inputs, false, false, false);

    auto fragment_inputs_str = fragment_inputs.createInterfaceBlockString(true);
    builders[0]->replaceFirstTag("FragmentShaderInputs", fragment_inputs_str);
    builders[1]->replaceFirstTag("FragmentShaderInputs", fragment_inputs_str);

    inject_into_vertex_shader(*builders[0],
                              type_info_ss,
                              vertex_attr_str,
                              prop_getter_str,
                              ubo_struct_builder.createStructString());
    inject_into_fragment_shader(*builders[1], num_samples_, false);
  }

  auto caches_points = shader_mgr.createCacheVector(std::move(builders), true);

  // Geometry shader billboards
  builders = shader_mgr.createBuilderVector({{"Marks/fastSymbolTemplate_passthru.vert"},
                                             {"Marks/fastSymbolTemplate.frag"},
                                             {"Marks/fastSymbolTemplate.geom"}});
  {
    auto [type_info_ss, vertex_attr_ss, prop_getter_ss, ubo_struct_builder] =
        create_render_prop_injectors(geometry_render_properties,
                                     "FAST_SYMBOL_VERT_UBO_TYPE");

    auto vertex_attr_str = vertex_attr_ss.str();
    auto prop_getter_str = prop_getter_ss.str();

    gfx::GlslStructBuilder fragment_inputs("FragmentShaderInputs");
    auto geometry_inputs = std::optional<gfx::GlslStructBuilder>{"GeometryShaderInputs"};
    generate_fast_symbol_interface_blocks(
        fragment_inputs, geometry_inputs, false, false, false);

    builders[0]->replaceFirstTag(
        "GeometryShaderInputs",
        geometry_inputs->createInterfaceBlockString(true, std::nullopt, false));
    builders[2]->replaceFirstTag(
        "GeometryShaderInputs",
        geometry_inputs->createInterfaceBlockString(true, std::nullopt, true));

    auto fragment_inputs_str = fragment_inputs.createInterfaceBlockString(true);
    builders[1]->replaceFirstTag("FragmentShaderInputs", fragment_inputs_str);
    builders[2]->replaceFirstTag("FragmentShaderInputs", fragment_inputs_str);

    inject_into_vertex_shader(*builders[0],
                              type_info_ss,
                              vertex_attr_str,
                              prop_getter_str,
                              ubo_struct_builder.createStructString());
    inject_into_fragment_shader(*builders[1], num_samples_, true);
    inject_into_geometry_shader(*builders[2]);
  }
  auto caches_geom = shader_mgr.createCacheVector(std::move(builders), true);

  //
  // Create materials
  //
  material_points_ = resource_mgr.createMaterial("Point", caches_points);
  material_geom_ = resource_mgr.createMaterial("Geom", caches_geom);

  material_points_->updateDescriptorSets();
  material_geom_->updateDescriptorSets();
  updateSymbolDefinitionUniforms();

  //
  // Create PrimitiveAssemblies
  //
  VboAttrToShaderAttrPairs attr_pairs = {{"x", "x"},
                                         {"y", "y"},
                                         {"width", "width"},
                                         {"height", "height"},
                                         {"shape", "shape"}};
  for (auto& data : distribution_data_) {
    PrimitiveAssemblyAttrInfo attr_info = {
        {static_cast<VertexBuffer*>(data.vbo.get()), buffer_layout_}, attr_pairs};

    data.primitive_assembly = resource_mgr.createPrimitiveAssembly(
        data.name + "point", PrimitiveTopology::kPointList, *material_points_, attr_info);

    data.primitive_assembly_geom = resource_mgr.createPrimitiveAssembly(
        data.name + "geom", PrimitiveTopology::kPointList, *material_geom_, attr_info);
  }

  //
  // Create pipelines
  //

  PipelineDescriptor pipeline_desc;
  pipeline_desc.setRasterSampleCount(raster_sample_count_);

  // Points
  pipeline_points_ = resource_mgr.createGraphicsPipeline(
      "Points",
      *material_points_,
      pipeline_desc,
      distribution_data_[Distribution::kGrid].primitive_assembly.get());
  pipeline_points_->create(*render_pass_);

  // Geometry shader
  pipeline_geom_ = resource_mgr.createGraphicsPipeline(
      "Geom",
      *material_geom_,
      pipeline_desc,
      distribution_data_[Distribution::kGrid].primitive_assembly_geom.get());
  pipeline_geom_->create(*render_pass_);
}

void SymbolSandbox::appInit() {
  QueryRenderer::init_symbol_defs(g_symbol_defs);

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
                                                Background::Type::kGridLines,
                                                Background::Type::kSymbolEdit};
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

void SymbolSandbox::initVertexBuffers() {
  buffer_layout_ = std::make_shared<InterleavedBufferLayout>();
  CHECK(buffer_layout_);
  buffer_layout_->addAttribute("x", BufferAttrType::kFloat);
  buffer_layout_->addAttribute("y", BufferAttrType::kFloat);
  buffer_layout_->addAttribute("width", BufferAttrType::kFloat);
  buffer_layout_->addAttribute("height", BufferAttrType::kFloat);
  buffer_layout_->addAttribute("shape", BufferAttrType::kUint);
  // TODO: random angle support
  // buffer_layout_->addAttribute("angle", BufferAttrType::kFloat);

  //
  // Populate DistributionData and create vertex buffers
  //
  updateSingleDistribution();
  updateGridDistribution();
  updateRandomDistribution();
}

void SymbolSandbox::createDistributionBuffer(DistributionData& data,
                                             const std::vector<SymbolVertex>& vertices,
                                             const std::string& name) {
  data.name = name;
  data.num_symbols = vertices.size();
  auto buffer_size = data.num_symbols * sizeof(SymbolVertex);
  data.vbo = device_->getResourceManager().createBuffer(
      name + " VBO",
      {BufferType::kVertexBuffer, buffer_size, BufferUsageBits::kLayoutBufferBit});

  data.vbo->updateSubDataWithLayout(vertices.data(), buffer_size, 0, buffer_layout_);
}

void SymbolSandbox::updateSymbolDefinitionUniforms() {
  auto update_material = [](Material& material) {
    material.setUniformAttribute("uSymbolFlags", g_symbol_defs.flags);
    material.setUniformAttribute("uVertCounts", g_symbol_defs.segment_counts);
    material.setUniformAttribute("uVertOffsets", g_symbol_defs.vert_offsets);
    material.setUniformAttribute("uSymbolVerts", g_symbol_defs.verts);
  };
  update_material(*material_points_);
  update_material(*material_geom_);
}

inline void apply_eccentricity(float& w, float& h, const float eccentricity) {
  if (eccentricity > 0.0f) {
    w *= 1.0f - eccentricity;
  } else if (eccentricity < 0.0f) {
    h *= 1.0f + eccentricity;
  }
}

void SymbolSandbox::updateSingleDistribution() {
  auto& data = distribution_data_[Distribution::kSingle];
  data.num_symbols = 1;

  float fwidth = static_cast<float>(window_width_) * 0.5f;
  float fheight = static_cast<float>(window_height_) * 0.5f;
  float aspect = fwidth / fheight;  // image aspect

  SymbolVertex v = {};
  v.x = fwidth;
  v.y = fheight;
  v.width = fwidth;
  v.height = fheight * aspect;
  apply_eccentricity(v.width, v.height, eccentricity_);
  v.shape = static_cast<uint32_t>(single_symbol_type_);

  if (data.vbo) {
    data.vbo->updateSubData(&v, sizeof(SymbolVertex), 0);
  } else {
    createDistributionBuffer(data, std::vector<SymbolVertex>{v}, "Single");
  }
  single_distribution_dirty_ = false;
}

void SymbolSandbox::updateGridDistribution() {
  auto& data = distribution_data_[Distribution::kGrid];
  data.num_symbols = static_cast<uint32_t>(SymbolShapeType::kCOUNT);

  std::vector<SymbolVertex> vertex_data(data.num_symbols);

  float fwindow_width = static_cast<float>(window_width_);
  float fwindow_height = static_cast<float>(window_height_);
  float aspect = fwindow_width / fwindow_height;
  float width = fwindow_width / static_cast<float>(kGridColumns) * 0.5f;
  float height = fwindow_height / static_cast<float>(kGridColumns) * 0.5f * aspect;
  apply_eccentricity(width, height, eccentricity_);

  uint32_t symbol_id = 0;
  float fx_step = 1.0f / kGridColumns;
  float fy_step = 1.0f / kGridRows;
  float fy = fy_step * 0.5f;

  for (uint32_t y = 0; y < kGridRows; y++, fy += fy_step) {
    float fx = fx_step * 0.5f;
    for (uint32_t x = 0; x < kGridColumns && symbol_id < data.num_symbols;
         x++, fx += fx_step, symbol_id++) {
      auto& vertex = vertex_data[symbol_id];
      vertex.x = fx * fwindow_width;
      vertex.y = fy * fwindow_height;
      vertex.width = width;
      vertex.height = height;
      vertex.shape = symbol_id;
    }
  }
  if (data.vbo) {
    data.vbo->updateSubData(
        vertex_data.data(), vertex_data.size() * sizeof(SymbolVertex), 0);

  } else {
    createDistributionBuffer(
        distribution_data_[Distribution::kGrid], vertex_data, "Grid");
  }

  grid_distribution_dirty_ = false;
}

void SymbolSandbox::updateRandomDistribution() {
  auto& data = distribution_data_[Distribution::kRandom];
  data.num_symbols = kMaxRandomSymbols;

  std::mt19937 p_gen;
  p_gen.seed(1);
  std::mt19937 s_gen;
  s_gen.seed(2);
  std::mt19937 a_gen;
  a_gen.seed(4);

  // Position distribution is 0 to 1
  // VPmatrix handles change to -1 to 1
  std::uniform_real_distribution<float> p_dis(0.0f, 1.0f);
  std::uniform_real_distribution<float> s_dis(0.0f, 1.0f);

  std::vector<SymbolVertex> vertex_data(data.num_symbols);

  float size_range = max_size_ - min_size_;
  for (uint32_t i = 0; i < data.num_symbols; i++) {
    auto& vertex = vertex_data[i];
    vertex.x = p_dis(p_gen) * static_cast<float>(window_width_);
    vertex.y = p_dis(p_gen) * static_cast<float>(window_height_);
    vertex.width = s_dis(s_gen) * size_range + min_size_;
    vertex.height = s_dis(s_gen) * size_range + min_size_;
    vertex.shape = i % static_cast<uint32_t>(SymbolShapeType::kCOUNT);
  }
  if (data.vbo) {
    data.vbo->updateSubData(
        vertex_data.data(), vertex_data.size() * sizeof(SymbolVertex), 0);
  } else {
    createDistributionBuffer(
        distribution_data_[Distribution::kRandom], vertex_data, "Random");
  }

  random_distribution_dirty_ = false;
}

void SymbolSandbox::maybeResetTime() {
  // Reset timing stats if anything changed that would invalidate current totals
  if ((distribution_ == Distribution::kSingle && single_distribution_dirty_) ||
      (distribution_ == Distribution::kGrid && grid_distribution_dirty_) ||
      (distribution_ == Distribution::kRandom && random_distribution_dirty_) ||
      uniforms_dirty_ || view_uniforms_dirty_ || render_time_dirty_ ||
      symbol_defs_dirty_) {
    num_renders_ = 0;
    total_time_ms_ = 0.0;
    render_time_dirty_ = false;
  }
}

void SymbolSandbox::updateUniforms() {
  if (symbol_defs_dirty_) {
    updateSymbolDefinitionUniforms();
    symbol_defs_dirty_ = false;
  }

  if (uniforms_dirty_) {
    Material& material = use_geometry_shader_ ? *material_geom_ : *material_points_;
    material.setUniformAttribute("opacity", opacity_);
    material.setUniformAttribute("fillColor", fill_color_);
    material.setUniformAttribute("fillOpacity", fill_opacity_);
    material.setUniformAttribute("strokeColor", stroke_color_);
    material.setUniformAttribute("strokeOpacity", stroke_opacity_);
    material.setUniformAttribute("strokeWidth", stroke_width_);

    material.setUniformAttribute("uPivotx", 0.0f);
    material.setUniformAttribute("uPivoty", 0.0f);

    if (use_geometry_shader_) {
      material.setUniformAttribute("uInvViewportWidth", 1.0f / (float)window_width_);
      material.setUniformAttribute("uInvViewportHeight", 1.0f / (float)window_height_);
      material.setUniformAttribute("angleUnit", 0);
      material.setUniformAttribute("angle", angle_degrees_);

      float angle = angle_degrees_ * 3.14159265359f / 180.0f;
      material.setUniformAttribute("uSinAngle", std::sin(angle));
      material.setUniformAttribute("uCosAngle", std::cos(angle));
    }

    uniforms_dirty_ = false;
  }

  if (view_uniforms_dirty_) {
    auto fwidth = static_cast<float>(window_width_);
    auto fheight = static_cast<float>(window_height_);
    auto proja = 2.0f / fwidth;
    auto projb = 2.0f / fheight;
    gfx::Math::Matrix2d<float> tmp = {{{proja, 0, 0, projb, -1, -1}}};

    material_points_->setUniformAttribute("uVPmatrix", tmp.getDataArrayRef());
    material_geom_->setUniformAttribute("uVPmatrix", tmp.getDataArrayRef());
    material_points_->setViewportAttributes(0, 0, window_width_, window_height_);

    view_uniforms_dirty_ = false;
  }
}

void SymbolSandbox::render() {
  maybeResetTime();

  // Update distribution vertex buffer
  if (distribution_ == Distribution::kSingle && single_distribution_dirty_) {
    updateSingleDistribution();
  }
  if (distribution_ == Distribution::kGrid && grid_distribution_dirty_) {
    updateGridDistribution();
  }
  if (distribution_ == Distribution::kRandom && random_distribution_dirty_) {
    updateRandomDistribution();
  }

  updateUniforms();

  // Draw points
  auto& data = distribution_data_[distribution_];
  auto& pipeline = use_geometry_shader_ ? *pipeline_geom_ : *pipeline_points_;
  uint32_t draw_count = distribution_ == Distribution::kRandom
                            ? static_cast<uint32_t>(random_draw_count_)
                            : data.num_symbols;

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

  // Draw the symbols
  auto clock_start = timer_start();
  cmd_list.beginRenderPass(*render_pass, *framebuffer_)
      .drawVertices(pipeline, *static_cast<VertexBuffer*>(data.vbo.get()), draw_count)
      .endRenderPass()
      .flush("Draw");

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

void SymbolSandbox::drawUI() {
  imgui_bridge_->beginRecording();

  //
  // Main panel
  //
  if (show_ui_) {
    imgui_bridge_->setNextWindowPos(ImGuiBridge::kTopLeft, 5.0f, ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(640, 800), ImGuiCond_Once);
    if (ImGui::Begin("SymbolSandbox", &show_ui_)) {
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

            // Single symbol combo box
            {
              ImGuiBridge::ScopedDisable guard(distribution_ == Distribution::kSingle);
              ImGui::PushItemWidth(300.0f);
              if (ImGuiBridge::doComboBox<SymbolShapeType>(
                      "##Shape", g_shape_strings, single_symbol_type_)) {
                single_distribution_dirty_ = true;
                background_->setSymbolFlagBits(
                    g_symbol_defs.flags[static_cast<int>(single_symbol_type_)]);
              }
              ImGui::PopItemWidth();
            }

            // Grid mode
            if (ImGui::RadioButton("Grid", distribution_ == Distribution::kGrid)) {
              distribution_ = Distribution::kGrid;
              render_time_dirty_ = true;
            }

            ImGui::Spacing();

            // Random mode
            if (ImGui::RadioButton("Random", distribution_ == Distribution::kRandom)) {
              distribution_ = Distribution::kRandom;
              render_time_dirty_ = true;
            }

            {  // Random distribution controls
              ImGuiBridge::ScopedDisable guard(distribution_ == Distribution::kRandom);
              if (ImGui::SliderInt(
                      "# Points", &random_draw_count_, 1, kMaxRandomSymbols)) {
                render_time_dirty_ = true;
              }
              if (random_draw_count_ < 1) {
                random_draw_count_ = 1;
              }
              ImGui::Spacing();
              random_distribution_dirty_ =
                  ImGui::DragFloatRange2("Size",
                                         &min_size_,
                                         &max_size_,
                                         0.25f,
                                         0.01f,
                                         100.0f,
                                         "Min: %.2f",
                                         "Max: %.2f",
                                         ImGuiSliderFlags_AlwaysClamp);
              imgui_bridge_->HelpMarker(
                  "left-click and drag left/right to change value\n"
                  "ctrl + left-click to type value");
            }
          }

          //
          // Properties
          //
          if (ImGui::CollapsingHeader("Properties", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Checkbox("Use Geometry Shader", &use_geometry_shader_)) {
              uniforms_dirty_ = true;
            }
            {  // Geometry shader controls
              ImGuiBridge::ScopedDisable guard(use_geometry_shader_);
              uniforms_dirty_ |=
                  ImGui::SliderFloat("Angle", &angle_degrees_, 0.0f, 360.0f);
            }

            ImGui::Separator();
            {
              ImGuiBridge::ScopedDisable guard(distribution_ == Distribution::kSingle ||
                                               distribution_ == Distribution::kGrid);
              bool changed =
                  ImGui::SliderFloat("Eccentricity", &eccentricity_, -1.0f, 1.0f);
              imgui_bridge_->HelpMarker(
                  "Values < 0 will squash height\nValues > 0 will squash width");
              single_distribution_dirty_ |= changed;
              grid_distribution_dirty_ |= changed;
            }
            ImGui::Separator();

            static constexpr ImGuiColorEditFlags color_edit_flags =
                ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel |
                ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_AlphaBar;
            uniforms_dirty_ |=
                ImGui::ColorEdit4("Fill Color", (float*)&fill_color_, color_edit_flags);
            uniforms_dirty_ |=
                ImGui::SliderFloat("Fill Opacity", &fill_opacity_, 0.0f, 1.0f);
            ImGui::Separator();
            uniforms_dirty_ |= ImGui::ColorEdit4(
                "Stroke Color", (float*)&stroke_color_, color_edit_flags);
            uniforms_dirty_ |=
                ImGui::SliderFloat("Stroke Opacity", &stroke_opacity_, 0.0f, 1.0f);
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
        static HelpInputTable hotkeys = {{{"", "D", "Cycle distribution"},
                                          {"", "S", "Single-symbol selection cycle"},
                                          {"", "G", "Use geometry shader"}},
                                         {{"", "R", "Reload shaders"}},
                                         {{"", "T", "Show time overlay"},
                                          {"", "U", "Show UI"},
                                          {"", "E", "Symbol Editor"}}};

        if (ImGui::BeginTabItem("Help")) {
          ImGui::CollapsingHeader("Hotkeys", ImGuiTreeNodeFlags_Leaf);
          draw_hotkey_help(hotkeys, true, false);
          ImGui::CollapsingHeader("Editor", ImGuiTreeNodeFlags_Leaf);
          ImGui::Text("%s", editor_help_text.c_str());

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

  if (show_editor_) {
    drawSymbolEditor();
  }

  imgui_bridge_->endRecordingAndDraw(*framebuffer_);
}

//
// Symbol editor
//
void SymbolSandbox::drawSymbolEditor() {
  imgui_bridge_->setNextWindowPos(ImGuiBridge::kTopRight, 5.0f, ImGuiCond_Once);
  if (show_editor_) {
    // Buttons
    ImGui::Begin("Editor", &show_editor_, ImGuiWindowFlags_AlwaysAutoResize);
    if (ImGui::Button("Set Edit Mode")) {
      background_->setType(Background::Type::kSymbolEdit);
      background_->setSymbolFlagBits(
          g_symbol_defs.flags[static_cast<int>(single_symbol_type_)]);
      distribution_ = Distribution::kSingle;
      render_time_dirty_ = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset Symbols")) {
      g_symbol_defs.resize(0);
      init_symbol_defs(g_symbol_defs);
      symbol_defs_dirty_ = true;
    }

    imgui_bridge_->HelpMarker(editor_help_text, true, 50);

    // Symbol selector
    ImGui::PushItemWidth(300.0f);
    if (ImGuiBridge::doComboBox<SymbolShapeType>(
            "##Shape", g_shape_strings, single_symbol_type_)) {
      single_distribution_dirty_ = true;
      background_->setSymbolFlagBits(
          g_symbol_defs.flags[static_cast<int>(single_symbol_type_)]);
    }
    ImGui::PopItemWidth();

    // Vertex table
    static constexpr ImGuiTableFlags flags =
        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
    const int kNumColumns = 2;
    const int kColumnWidth = 200;
    if (ImGui::BeginTable("Symbol Definition", kNumColumns, flags)) {
      ImGui::TableSetupColumn("X", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Y", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableHeadersRow();

      int index = static_cast<int>(single_symbol_type_);
      int num_verts = g_symbol_defs.segment_counts[index] + 1;
      int first_vert = g_symbol_defs.vert_offsets[index];
      auto& verts = g_symbol_defs.verts;
      for (int row = 0; row < num_verts; row++) {
        ImGui::TableNextRow();
        for (int column = 0; column < kNumColumns; column++) {
          ImGui::TableSetColumnIndex(column);
          ImGui::PushID(row * kNumColumns + column);  // assign unique id
          ImGui::PushItemWidth(kColumnWidth);
          if (ImGui::SliderFloat(
                  "##v", &verts[first_vert + row][column], -1.0f, 1.0f, "%.3f")) {
            symbol_defs_dirty_ = true;
          }
          ImGui::PopItemWidth();
          ImGui::PopID();
        }
      }
      ImGui::EndTable();
    }
    ImGui::End();
  }
}

void SymbolSandbox::destroyMaterialsAndPipelines() {
  auto& resource_mgr = device_->getResourceManager();
  if (pipeline_points_) {
    resource_mgr.destroyPipeline(std::move(pipeline_points_));
  }
  if (pipeline_geom_) {
    resource_mgr.destroyPipeline(std::move(pipeline_geom_));
  }
  for (auto& data : distribution_data_) {
    data.primitive_assembly = nullptr;
  }

  material_points_ = nullptr;
  material_geom_ = nullptr;
}

void SymbolSandbox::appShutdown() {
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

void SymbolSandbox::reloadShaders() {
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

void SymbolSandbox::handleWSIEvent(const WSIEvent& event) {
  if (event.type() == WSIEvent::Type::kWindowResize) {
    handleWindowResizeEvent(static_cast<const WSIWindowResizeEvent&>(event));
  } else if (event.type() == WSIEvent::Type::kKeyboard) {
    handleKeyboardEvent(static_cast<const WSIKeyboardEvent&>(event));
  }
}

void SymbolSandbox::handleWindowResizeEvent(const WSIWindowResizeEvent& event) {
  single_distribution_dirty_ = true;
  grid_distribution_dirty_ = true;
  random_distribution_dirty_ = true;
  uniforms_dirty_ = true;
  view_uniforms_dirty_ = true;
  background_->setImageSize(event.width(), event.height());
}

void SymbolSandbox::handleKeyboardEvent(const WSIKeyboardEvent& event) {
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

      // Cycle single symbol type
      case WSIKeyboardKey::kS: {
        size_t type = static_cast<size_t>(single_symbol_type_) + 1;
        if (type >= g_shape_strings.size()) {
          type = 0;
        }
        single_symbol_type_ = static_cast<SymbolShapeType>(type);
        single_distribution_dirty_ = true;
      } break;

      // Toggle geometry shader
      case WSIKeyboardKey::kG:
        use_geometry_shader_ = !use_geometry_shader_;
        uniforms_dirty_ = true;
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

      // Toggle symbol editor
      case WSIKeyboardKey::kE:
        show_editor_ = !show_editor_;
        break;

      default:
        break;
    }
  }
}

DECLARE_WSI_APP_MAIN(SymbolSandbox)
