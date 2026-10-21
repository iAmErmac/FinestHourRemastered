#include "gpu/vulkan/sampler.h"

#include <algorithm>
#include <cstring>

namespace gpu::vulkan {
namespace {

float Bits(uint32_t value) {
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

VkFilter Filter(uint32_t value) { return value >= 2 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST; }

VkSamplerAddressMode Address(uint32_t value) {
    switch (value) {
        case 2: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        case 3: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        case 5: return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
        case 4: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        default: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    }
}

}  // namespace

VkSamplerCreateInfo MakeSamplerCreateInfo(const std::array<uint32_t, kTssCount>& state, float max_anisotropy,
                                          float max_lod_bias) {
    VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    info.magFilter = Filter(state[kTssMagFilter]);
    info.minFilter = Filter(state[kTssMinFilter]);
    info.mipmapMode = state[kTssMipFilter] == 2 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    info.addressModeU = Address(state[kTssAddressU]);
    info.addressModeV = Address(state[kTssAddressV]);
    info.addressModeW = Address(state[kTssAddressW]);
    const float lod_bias_limit = std::max(0.0f, max_lod_bias);
    info.mipLodBias = std::clamp(Bits(state[kTssMipLodBias]), -lod_bias_limit, lod_bias_limit);
    info.minLod = float(state[kTssMaxMipLevel]);
    info.maxLod = state[kTssMipFilter] ? VK_LOD_CLAMP_NONE : info.minLod;
    info.borderColor = state[kTssBorderColor] == 0xFFFFFFFF ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE
                    : state[kTssBorderColor] == 0xFF000000 ? VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK
                                                            : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    const bool anisotropic = state[kTssMinFilter] == 3 || state[kTssMagFilter] == 3;
    if (anisotropic && state[kTssMaxAnisotropy] > 1) {
        info.anisotropyEnable = VK_TRUE;
        info.maxAnisotropy = std::min(float(state[kTssMaxAnisotropy]), std::max(1.0f, max_anisotropy));
    }
    return info;
}

}  // namespace gpu::vulkan
