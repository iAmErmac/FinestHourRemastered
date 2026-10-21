#include "gpu/vulkan/pipeline.h"

#include <algorithm>
#include <array>
#include <bit>

namespace gpu::vulkan {
namespace {

VkShaderStageFlagBits VkStage(ShaderStage stage) {
    switch (stage) {
        case ShaderStage::Vertex: return VK_SHADER_STAGE_VERTEX_BIT;
        case ShaderStage::Pixel: return VK_SHADER_STAGE_FRAGMENT_BIT;
        case ShaderStage::Geometry: return VK_SHADER_STAGE_GEOMETRY_BIT;
        case ShaderStage::Hull: return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
        case ShaderStage::Domain: return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    }
    return VK_SHADER_STAGE_FLAG_BITS_MAX_ENUM;
}

bool InRange(ShaderBinding::Kind kind, uint32_t binding) {
    switch (kind) {
        case ShaderBinding::Kind::Constant: return binding < kConstantBindingCount;
        case ShaderBinding::Kind::Texture: return binding >= kTextureBindingBase && binding < kTextureBindingBase + kTextureBindingCount;
        case ShaderBinding::Kind::Sampler: return binding >= kSamplerBindingBase && binding < kSamplerBindingBase + kSamplerBindingCount;
    }
    return false;
}

}  // namespace

GraphicsPipeline::~GraphicsPipeline() { Reset(); }

void GraphicsPipeline::Reset() {
    if (device_ != VK_NULL_HANDLE) {
        if (pipeline_ && destroy_pipeline_) destroy_pipeline_(device_, pipeline_, nullptr);
        if (layout_ && destroy_layout_) destroy_layout_(device_, layout_, nullptr);
    }
    device_ = VK_NULL_HANDLE;
    pipeline_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE;
    contract_ = {};
    color_format_ = VK_FORMAT_UNDEFINED;
    depth_format_ = VK_FORMAT_UNDEFINED;
    samples_ = VK_SAMPLE_COUNT_1_BIT;
    destroy_pipeline_ = nullptr;
    destroy_layout_ = nullptr;
    destroy_shader_module_ = nullptr;
    begin_rendering_ = nullptr;
    end_rendering_ = nullptr;
    bind_pipeline_ = nullptr;
    set_viewport_ = nullptr;
    set_scissor_ = nullptr;
    draw_ = nullptr;
}

bool GraphicsPipeline::ValidateShader(const ShaderModule* module, ShaderStage expected, const char* name,
                                      std::string& error) const {
    if (!module || module->spirv.empty()) {
        error = std::string("Vulkan pipeline requires a non-empty ") + name + " shader module";
        return false;
    }
    if (module->reflection.stage != expected || module->reflection.entry.empty()) {
        error = std::string("Vulkan pipeline ") + name + " shader reflection has the wrong stage or entry point";
        return false;
    }
    if (!(contract_.stages & VkStage(expected))) {
        error = std::string("Vulkan pipeline ") + name + " stage is not allowed by pass contract " + contract_.name;
        return false;
    }
    for (const ShaderBinding& binding : module->reflection.bindings) {
        if (binding.set != kDescriptorSetIndex || !InRange(binding.kind, binding.binding)) {
            error = std::string("Vulkan pipeline ") + name + " shader reflection is outside the fixed descriptor layout";
            return false;
        }
    }
    return true;
}

bool GraphicsPipeline::Init(const Context& context, const DescriptorArena& descriptors,
                            const GraphicsPipelineDesc& description, std::string& error) {
    Reset();
    error.clear();
    if (!context.Initialized() || !descriptors.Layout()) {
        error = "Vulkan graphics pipeline requires initialized context and descriptor layout";
        return false;
    }
    if (descriptors.Device() != context.Device()) {
        error = "Vulkan graphics pipeline context does not own the supplied descriptor layout";
        return false;
    }
    if (description.contract.pass != GraphicsPass::SampledImage || description.contract.color_attachment_count != 1 ||
        description.contract.has_depth || !description.vertex || !description.fragment || description.geometry ||
        description.hull || description.domain || description.color_format == VK_FORMAT_UNDEFINED ||
        description.depth_format != VK_FORMAT_UNDEFINED || description.samples != VK_SAMPLE_COUNT_1_BIT ||
        description.topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST) {
        error = "Vulkan sampled-image pipeline contract has unsupported stages, attachments or fixed-function state";
        return false;
    }
    device_ = context.Device();
    contract_ = description.contract;
    color_format_ = description.color_format;
    depth_format_ = description.depth_format;
    samples_ = description.samples;
    if (!ValidateShader(description.vertex, ShaderStage::Vertex, "vertex", error) ||
        !ValidateShader(description.fragment, ShaderStage::Pixel, "fragment", error)) {
        Reset();
        return false;
    }

    destroy_pipeline_ = reinterpret_cast<PFN_vkDestroyPipeline>(context.DeviceProc("vkDestroyPipeline"));
    destroy_layout_ = reinterpret_cast<PFN_vkDestroyPipelineLayout>(context.DeviceProc("vkDestroyPipelineLayout"));
    destroy_shader_module_ = reinterpret_cast<PFN_vkDestroyShaderModule>(context.DeviceProc("vkDestroyShaderModule"));
    begin_rendering_ = reinterpret_cast<PFN_vkCmdBeginRendering>(context.DeviceProc("vkCmdBeginRendering"));
    end_rendering_ = reinterpret_cast<PFN_vkCmdEndRendering>(context.DeviceProc("vkCmdEndRendering"));
    bind_pipeline_ = reinterpret_cast<PFN_vkCmdBindPipeline>(context.DeviceProc("vkCmdBindPipeline"));
    set_viewport_ = reinterpret_cast<PFN_vkCmdSetViewport>(context.DeviceProc("vkCmdSetViewport"));
    set_scissor_ = reinterpret_cast<PFN_vkCmdSetScissor>(context.DeviceProc("vkCmdSetScissor"));
    draw_ = reinterpret_cast<PFN_vkCmdDraw>(context.DeviceProc("vkCmdDraw"));
    auto create_layout = reinterpret_cast<PFN_vkCreatePipelineLayout>(context.DeviceProc("vkCreatePipelineLayout"));
    auto create_shader = reinterpret_cast<PFN_vkCreateShaderModule>(context.DeviceProc("vkCreateShaderModule"));
    auto create_pipeline = reinterpret_cast<PFN_vkCreateGraphicsPipelines>(context.DeviceProc("vkCreateGraphicsPipelines"));
    if (!destroy_pipeline_ || !destroy_layout_ || !destroy_shader_module_ || !begin_rendering_ || !end_rendering_ ||
        !bind_pipeline_ || !set_viewport_ || !set_scissor_ || !draw_ || !create_layout || !create_shader ||
        !create_pipeline) {
        error = "Vulkan graphics pipeline entry points are unavailable";
        Reset();
        return false;
    }

    VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkDescriptorSetLayout set_layout = descriptors.Layout();
    layout.setLayoutCount = 1;
    layout.pSetLayouts = &set_layout;
    if (create_layout(device_, &layout, nullptr, &layout_) != VK_SUCCESS) {
        error = "vkCreatePipelineLayout failed";
        Reset();
        return false;
    }

    VkShaderModule modules[2] = {};
    const ShaderModule* sources[2] = {description.vertex, description.fragment};
    for (uint32_t i = 0; i < 2; ++i) {
        VkShaderModuleCreateInfo create{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        create.codeSize = sources[i]->spirv.size() * sizeof(uint32_t);
        create.pCode = sources[i]->spirv.data();
        if (create_shader(device_, &create, nullptr, &modules[i]) != VK_SUCCESS) {
            error = "vkCreateShaderModule failed for the sampled-image pipeline";
            for (VkShaderModule module : modules)
                if (module) destroy_shader_module_(device_, module, nullptr);
            Reset();
            return false;
        }
    }

    std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, modules[0],
                 description.vertex->reflection.entry.c_str(), nullptr};
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, modules[1],
                 description.fragment->reflection.entry.c_str(), nullptr};
    VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = description.topology;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = description.samples;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                                      VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments = &blend_attachment;
    const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamic_states;
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &color_format_;
    rendering.depthAttachmentFormat = VK_FORMAT_UNDEFINED;
    rendering.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
    VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeline.pNext = &rendering;
    pipeline.stageCount = static_cast<uint32_t>(stages.size());
    pipeline.pStages = stages.data();
    pipeline.pVertexInputState = &vertex_input;
    pipeline.pInputAssemblyState = &assembly;
    pipeline.pViewportState = &viewport;
    pipeline.pRasterizationState = &raster;
    pipeline.pMultisampleState = &multisample;
    pipeline.pDepthStencilState = &depth;
    pipeline.pColorBlendState = &blend;
    pipeline.pDynamicState = &dynamic;
    pipeline.layout = layout_;
    const VkResult result = create_pipeline(device_, VK_NULL_HANDLE, 1, &pipeline, nullptr, &pipeline_);
    for (VkShaderModule module : modules)
        if (module) destroy_shader_module_(device_, module, nullptr);
    if (result != VK_SUCCESS) {
        error = "vkCreateGraphicsPipelines failed: VkResult " + std::to_string(static_cast<int>(result));
        Reset();
        return false;
    }
    return true;
}

bool GraphicsPipeline::RecordDraw(
    VkCommandBuffer command, const DescriptorArena& descriptors, const DescriptorSet& set,
    const std::array<uint32_t, kDynamicConstantBindingCount>& dynamic_offsets, const RenderTarget& target,
    uint32_t vertex_count, std::string& error) const {
    error.clear();
    if (!pipeline_ || !layout_ || descriptors.Device() != device_ || !command || !target.color ||
        target.color_format != color_format_ ||
        !target.extent.width || !target.extent.height || !vertex_count ||
        target.color_format == VK_FORMAT_UNDEFINED) {
        error = "Vulkan sampled-image draw has invalid pipeline, target or vertex count";
        return false;
    }
    if (!descriptors.Validate(set, dynamic_offsets, error)) return false;
    VkRenderingAttachmentInfo attachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    attachment.imageView = target.color;
    attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.clearValue = target.clear;
    VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
    rendering.renderArea = {{0, 0}, target.extent};
    rendering.layerCount = 1;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachments = &attachment;
    begin_rendering_(command, &rendering);
    bind_pipeline_(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    if (!descriptors.Bind(command, layout_, set, dynamic_offsets, error)) {
        end_rendering_(command);
        return false;
    }
    VkViewport viewport{0.0f, 0.0f, static_cast<float>(target.extent.width), static_cast<float>(target.extent.height), 0.0f,
                        1.0f};
    VkRect2D scissor{{0, 0}, target.extent};
    set_viewport_(command, 0, 1, &viewport);
    set_scissor_(command, 0, 1, &scissor);
    draw_(command, vertex_count, 1, 0, 0);
    end_rendering_(command);
    return true;
}

}  // namespace gpu::vulkan
