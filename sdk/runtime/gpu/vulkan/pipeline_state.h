#pragma once

#include "gpu/vulkan/context.h"
#include "gpu/vulkan/shader.h"

#include <array>
#include <cstdint>

namespace gpu::vulkan {

struct PipelineStateDesc {
    std::array<VkVertexInputBindingDescription, 16> bindings{};
    uint32_t binding_count = 0;
    std::array<VkVertexInputAttributeDescription, 16> attributes{};
    uint32_t attribute_count = 0;
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPolygonMode polygon_mode = VK_POLYGON_MODE_FILL;
    VkCullModeFlags cull_mode = VK_CULL_MODE_NONE;
    VkFrontFace front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    VkBool32 depth_bias_enable = VK_FALSE;
    float depth_bias_constant = 0.0f;
    float depth_bias_slope = 0.0f;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    VkBool32 depth_test_enable = VK_FALSE;
    VkBool32 depth_write_enable = VK_FALSE;
    VkCompareOp depth_compare = VK_COMPARE_OP_ALWAYS;
    VkBool32 stencil_test_enable = VK_FALSE;
    VkStencilOpState stencil_front{};
    VkStencilOpState stencil_back{};
    VkColorComponentFlags color_write_mask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                             VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkBool32 blend_enable = VK_FALSE;
    VkBlendFactor src_color_blend = VK_BLEND_FACTOR_ONE;
    VkBlendFactor dst_color_blend = VK_BLEND_FACTOR_ZERO;
    VkBlendOp color_blend_op = VK_BLEND_OP_ADD;
    VkBlendFactor src_alpha_blend = VK_BLEND_FACTOR_ONE;
    VkBlendFactor dst_alpha_blend = VK_BLEND_FACTOR_ZERO;
    VkBlendOp alpha_blend_op = VK_BLEND_OP_ADD;
    VkFormat color_format = VK_FORMAT_UNDEFINED;
    VkFormat depth_format = VK_FORMAT_UNDEFINED;
    VkFormat stencil_format = VK_FORMAT_UNDEFINED;
};

// Raw XDK render-state values consumed by the product's backend. Keeping this
// translation here lets focused GPU checks exercise the same mapping as draws.
struct GuestPipelineState {
    uint32_t cull_mode = 0;
    bool triangles = true;
    bool z_enable = false;
    bool z_write_enable = false;
    uint32_t z_func = 0;
    bool stencil_enable = false;
    uint32_t stencil_func = 0;
    uint32_t stencil_mask = 0;
    uint32_t stencil_write_mask = 0;
    uint32_t stencil_fail = 0;
    uint32_t stencil_zfail = 0;
    uint32_t stencil_pass = 0;
    uint32_t color_write_enable = 0x01010101u;
    bool blend_enable = false;
    uint32_t src_blend = 1;
    uint32_t dst_blend = 0;
    uint32_t blend_op = 0;
    bool solid_offset_enable = false;
    float depth_offset = 0.0f;
    float slope_scale = 0.0f;
    float zscale = 16777215.0f;
    VkFormat color_format = VK_FORMAT_UNDEFINED;
    VkFormat depth_format = VK_FORMAT_UNDEFINED;
    VkFormat stencil_format = VK_FORMAT_UNDEFINED;
};

VkFormat GuestVertexFormat(uint32_t format);

void TranslateGuestPipelineState(const GuestPipelineState& guest, PipelineStateDesc& state);

VkPipeline CreatePipelineState(const Context& context, VkPipelineLayout layout, const ShaderModule& vertex,
                               VkShaderModule vertex_module, const ShaderModule& pixel, VkShaderModule pixel_module,
                               const ShaderModule* geometry, VkShaderModule geometry_module,
                               const PipelineStateDesc& state);

}  // namespace gpu::vulkan
