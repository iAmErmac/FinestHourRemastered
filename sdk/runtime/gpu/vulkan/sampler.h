#pragma once

#include "gpu/state.h"
#include "gpu/vulkan/context.h"

#include <array>

namespace gpu::vulkan {

// Translate one XDK stage sampler state into the immutable Vulkan sampler description used by the backend and
// focused offscreen fixtures. The device limit is supplied by the caller because it is a physical-device property.
VkSamplerCreateInfo MakeSamplerCreateInfo(const std::array<uint32_t, kTssCount>& state, float max_anisotropy,
                                          float max_lod_bias);

// Own the extension node used for one custom sampler so the pNext target remains valid through vkCreateSampler.
struct SamplerCustomBorderColorInfo {
    VkSamplerCustomBorderColorCreateInfoEXT extension{VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT};
};

SamplerCustomBorderColorInfo MakeSamplerCustomBorderColorInfo(uint32_t argb, VkFormat format);

// The guest shadow format selects the normalization scale used by generated PixelHlsl.
uint8_t ShadowDepthKind(uint32_t xbox_format);

// D3D comparison samplers receive texel OP reference; Vulkan receives reference OP texel.
VkCompareOp ShadowCompareOp(uint32_t guest_compare);

}  // namespace gpu::vulkan
