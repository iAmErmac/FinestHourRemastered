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

SamplerCustomBorderColorInfo MakeSamplerCustomBorderColorInfo(uint32_t argb, VkFormat format) {
    SamplerCustomBorderColorInfo result;
    result.extension.format = format;
    result.extension.customBorderColor.float32[0] = float((argb >> 16) & 255) / 255.0f;
    result.extension.customBorderColor.float32[1] = float((argb >> 8) & 255) / 255.0f;
    result.extension.customBorderColor.float32[2] = float(argb & 255) / 255.0f;
    result.extension.customBorderColor.float32[3] = float((argb >> 24) & 255) / 255.0f;
    return result;
}

uint8_t ShadowDepthKind(uint32_t xbox_format) {
    // D16 and LIN_D16 are the only guest depth formats normalized by 65535; D24/F24 use the 24-bit path.
    return xbox_format == 0x2C || xbox_format == 0x30 ? 2 : 1;
}

VkCompareOp ShadowCompareOp(uint32_t guest_compare) {
    static constexpr uint8_t swap[8] = {0, 4, 2, 6, 1, 5, 3, 7};
    const uint32_t compare = guest_compare >= 0x200 && guest_compare <= 0x207 ? guest_compare - 0x200 : 7;
    return static_cast<VkCompareOp>(swap[compare]);
}

}  // namespace gpu::vulkan
