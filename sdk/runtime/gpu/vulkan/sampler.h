#pragma once

#include "gpu/state.h"
#include "gpu/vulkan/context.h"

#include <array>

namespace gpu::vulkan {

// Translate one XDK stage sampler state into the immutable Vulkan sampler description used by the backend and
// focused offscreen fixtures. The device limit is supplied by the caller because it is a physical-device property.
VkSamplerCreateInfo MakeSamplerCreateInfo(const std::array<uint32_t, kTssCount>& state, float max_anisotropy,
                                          float max_lod_bias);

}  // namespace gpu::vulkan
