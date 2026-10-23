#include "gpu/vulkan/pipeline_state.h"
#include "gpu/vulkan/raster.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <dxgiformat.h>

namespace gpu::vulkan {

namespace {

VkCompareOp Compare(uint32_t value) {
    return value >= 0x200 && value <= 0x207 ? VkCompareOp(value - 0x200) : VK_COMPARE_OP_ALWAYS;
}

VkStencilOp StencilOp(uint32_t value) {
    switch (value) {
        case 0: return VK_STENCIL_OP_ZERO;
        case 0x1e01: return VK_STENCIL_OP_REPLACE;
        case 0x1e02: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
        case 0x1e03: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
        case 0x150a: return VK_STENCIL_OP_INVERT;
        case 0x8507: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
        case 0x8508: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
        default: return VK_STENCIL_OP_KEEP;
    }
}

VkBlendFactor Blend(uint32_t value, bool alpha) {
    switch (value) {
        case 0: return VK_BLEND_FACTOR_ZERO;
        case 1: return VK_BLEND_FACTOR_ONE;
        case 0x300: return alpha ? VK_BLEND_FACTOR_SRC_ALPHA : VK_BLEND_FACTOR_SRC_COLOR;
        case 0x301: return alpha ? VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA : VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case 0x302: return VK_BLEND_FACTOR_SRC_ALPHA;
        case 0x303: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case 0x304: return VK_BLEND_FACTOR_DST_ALPHA;
        case 0x305: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case 0x306: return alpha ? VK_BLEND_FACTOR_DST_ALPHA : VK_BLEND_FACTOR_DST_COLOR;
        case 0x307: return alpha ? VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA : VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        case 0x308: return alpha ? VK_BLEND_FACTOR_ONE : VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
        case 0x8001: return VK_BLEND_FACTOR_CONSTANT_COLOR;
        case 0x8002: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
        case 0x8003: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
        case 0x8004: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
        default: return VK_BLEND_FACTOR_ONE;
    }
}

VkBlendOp BlendOp(uint32_t value) {
    switch (value) {
        case 0x800A: return VK_BLEND_OP_SUBTRACT;
        case 0x800B: return VK_BLEND_OP_REVERSE_SUBTRACT;
        case 0x8007: return VK_BLEND_OP_MIN;
        case 0x8008: return VK_BLEND_OP_MAX;
        default: return VK_BLEND_OP_ADD;
    }
}

}  // namespace

VkFormat GuestVertexFormat(uint32_t format) {
    switch (format) {
        case DXGI_FORMAT_R32G32B32A32_FLOAT: return VK_FORMAT_R32G32B32A32_SFLOAT;
        case DXGI_FORMAT_R32G32B32_FLOAT: return VK_FORMAT_R32G32B32_SFLOAT;
        case DXGI_FORMAT_R32G32_FLOAT: return VK_FORMAT_R32G32_SFLOAT;
        case DXGI_FORMAT_R32_FLOAT: return VK_FORMAT_R32_SFLOAT;
        case DXGI_FORMAT_R32_UINT: return VK_FORMAT_R32_UINT;
        case DXGI_FORMAT_R16G16B16A16_SNORM: return VK_FORMAT_R16G16B16A16_SNORM;
        case DXGI_FORMAT_R16G16B16A16_SINT: return VK_FORMAT_R16G16B16A16_SINT;
        case DXGI_FORMAT_R16G16_SNORM: return VK_FORMAT_R16G16_SNORM;
        case DXGI_FORMAT_R16G16_SINT: return VK_FORMAT_R16G16_SINT;
        case DXGI_FORMAT_R16_SNORM: return VK_FORMAT_R16_SNORM;
        case DXGI_FORMAT_R16_SINT: return VK_FORMAT_R16_SINT;
        case DXGI_FORMAT_R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
        case DXGI_FORMAT_R8G8_UNORM: return VK_FORMAT_R8G8_UNORM;
        case DXGI_FORMAT_R8_UNORM: return VK_FORMAT_R8_UNORM;
        case DXGI_FORMAT_B8G8R8A8_UNORM: return VK_FORMAT_B8G8R8A8_UNORM;
        default: return VK_FORMAT_UNDEFINED;
    }
}

void TranslateGuestPipelineState(const GuestPipelineState& guest, PipelineStateDesc& state) {
    // Preserve the vertex layout and topology already prepared for this draw.
    state.cull_mode = RasterCullMode(guest.cull_mode, guest.triangles);
    state.front_face = RasterFrontFace(guest.cull_mode);
    const bool depth_offset = guest.triangles && guest.solid_offset_enable;
    state.depth_bias_enable = depth_offset ? VK_TRUE : VK_FALSE;
    state.depth_bias_constant = depth_offset
                                    ? std::lround(guest.depth_offset * 16777215.0f /
                                                  std::max(guest.zscale, 1.0f)) /
                                          16777215.0f
                                    : 0.0f;
    state.depth_bias_slope = depth_offset ? guest.slope_scale : 0.0f;
    state.depth_test_enable = (guest.depth_format != VK_FORMAT_UNDEFINED && guest.z_enable) ? VK_TRUE : VK_FALSE;
    state.depth_write_enable = state.depth_test_enable && guest.z_write_enable ? VK_TRUE : VK_FALSE;
    state.depth_compare = Compare(guest.z_func);
    state.stencil_test_enable = (guest.depth_format != VK_FORMAT_UNDEFINED && guest.stencil_enable) ? VK_TRUE : VK_FALSE;
    state.stencil_front.compareOp = Compare(guest.stencil_func);
    state.stencil_front.compareMask = guest.stencil_mask & 0xff;
    state.stencil_front.writeMask = guest.stencil_write_mask & 0xff;
    state.stencil_front.failOp = StencilOp(guest.stencil_fail);
    state.stencil_front.depthFailOp = StencilOp(guest.stencil_zfail);
    state.stencil_front.passOp = StencilOp(guest.stencil_pass);
    state.stencil_back = state.stencil_front;
    const uint32_t color_write = guest.color_write_enable;
    state.color_write_mask = ((color_write >> 16) & 1) | (((color_write >> 8) & 1) << 1) |
                             (((color_write >> 0) & 1) << 2) | (((color_write >> 24) & 1) << 3);
    state.blend_enable = guest.blend_enable ? VK_TRUE : VK_FALSE;
    state.src_color_blend = Blend(guest.src_blend, false);
    state.dst_color_blend = Blend(guest.dst_blend, false);
    state.color_blend_op = BlendOp(guest.blend_op);
    state.src_alpha_blend = Blend(guest.src_blend, true);
    state.dst_alpha_blend = Blend(guest.dst_blend, true);
    state.alpha_blend_op = BlendOp(guest.blend_op);
    state.color_format = guest.color_format;
    state.depth_format = guest.depth_format;
    state.stencil_format = guest.stencil_format;
}

VkPipeline CreatePipelineState(const Context& context, VkPipelineLayout layout, const ShaderModule& vertex,
                               VkShaderModule vertex_module, const ShaderModule& pixel, VkShaderModule pixel_module,
                               const ShaderModule* geometry, VkShaderModule geometry_module,
                               const PipelineStateDesc& state) {
    auto create = reinterpret_cast<PFN_vkCreateGraphicsPipelines>(context.DeviceProc("vkCreateGraphicsPipelines"));
    if (!create || !layout || !vertex_module || !pixel_module || (geometry && !geometry_module) ||
        state.binding_count > state.bindings.size() || state.attribute_count > state.attributes.size())
        return VK_NULL_HANDLE;

    std::array<VkPipelineShaderStageCreateInfo, 3> stages{};
    uint32_t stage_count = 0;
    stages[stage_count++] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                             VK_SHADER_STAGE_VERTEX_BIT, vertex_module, vertex.reflection.entry.c_str(), nullptr};
    if (geometry)
        stages[stage_count++] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                                 VK_SHADER_STAGE_GEOMETRY_BIT, geometry_module, geometry->reflection.entry.c_str(), nullptr};
    stages[stage_count++] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                             VK_SHADER_STAGE_FRAGMENT_BIT, pixel_module, pixel.reflection.entry.c_str(), nullptr};

    VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertex_input.vertexBindingDescriptionCount = state.binding_count;
    vertex_input.pVertexBindingDescriptions = state.bindings.data();
    vertex_input.vertexAttributeDescriptionCount = state.attribute_count;
    vertex_input.pVertexAttributeDescriptions = state.attributes.data();
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = state.topology;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = state.polygon_mode;
    raster.cullMode = state.cull_mode;
    raster.frontFace = state.front_face;
    raster.depthBiasEnable = state.depth_bias_enable;
    raster.depthBiasConstantFactor = state.depth_bias_constant;
    raster.depthBiasSlopeFactor = state.depth_bias_slope;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = state.samples;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.depthTestEnable = state.depth_test_enable;
    depth.depthWriteEnable = state.depth_write_enable;
    depth.depthCompareOp = state.depth_compare;
    depth.stencilTestEnable = state.stencil_test_enable;
    depth.front = state.stencil_front;
    depth.back = state.stencil_back;
    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.colorWriteMask = state.color_write_mask;
    blend_attachment.blendEnable = state.blend_enable;
    blend_attachment.srcColorBlendFactor = state.src_color_blend;
    blend_attachment.dstColorBlendFactor = state.dst_color_blend;
    blend_attachment.colorBlendOp = state.color_blend_op;
    blend_attachment.srcAlphaBlendFactor = state.src_alpha_blend;
    blend_attachment.dstAlphaBlendFactor = state.dst_alpha_blend;
    blend_attachment.alphaBlendOp = state.alpha_blend_op;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = state.color_format == VK_FORMAT_UNDEFINED ? 0 : 1;
    blend.pAttachments = &blend_attachment;
    const std::array<VkDynamicState, 4> dynamic_states = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_BLEND_CONSTANTS,
        VK_DYNAMIC_STATE_STENCIL_REFERENCE};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
    dynamic.pDynamicStates = dynamic_states.data();
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = state.color_format == VK_FORMAT_UNDEFINED ? 0 : 1;
    rendering.pColorAttachmentFormats = rendering.colorAttachmentCount ? &state.color_format : nullptr;
    rendering.depthAttachmentFormat = state.depth_format;
    rendering.stencilAttachmentFormat = state.stencil_format;
    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    info.pNext = &rendering;
    info.stageCount = stage_count;
    info.pStages = stages.data();
    info.pVertexInputState = &vertex_input;
    info.pInputAssemblyState = &assembly;
    info.pViewportState = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pDepthStencilState = &depth;
    info.pColorBlendState = &blend;
    info.pDynamicState = &dynamic;
    info.layout = layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    return create(context.Device(), context.PipelineCacheHandle(), 1, &info, nullptr, &pipeline) == VK_SUCCESS ? pipeline : VK_NULL_HANDLE;
}

}  // namespace gpu::vulkan
