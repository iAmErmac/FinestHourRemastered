#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace gpu {

// Build the title ramp after applying the user's brightness curve. The output
// is RGB channel-major because the Vulkan blit ABI addresses channels directly.
// D3D12 reorders these same values into its BGRA LUT texture at upload time.
inline void BuildGammaLut(const uint8_t* title_ramp, uint32_t brightness, uint8_t* output, bool& effective) {
    uint8_t source[768]{};
    bool set = false;
    if (title_ramp) {
        for (uint32_t i = 0; i < 768; ++i) {
            source[i] = title_ramp[i];
            set = set || title_ramp[i] != 0;
        }
    }
    if (!set) {
        for (uint32_t i = 0; i < 256; ++i)
            source[i] = source[256 + i] = source[512 + i] = static_cast<uint8_t>(i);
    }

    const double exponent = 100.0 / std::max(brightness, 1u);
    auto curve = [exponent](uint8_t value) {
        return static_cast<uint8_t>(std::lround(255.0 * std::pow(value / 255.0, exponent)));
    };

    effective = false;
    for (uint32_t channel = 0; channel < 3; ++channel) {
        for (uint32_t index = 0; index < 256; ++index) {
            const uint8_t value = curve(source[channel * 256 + index]);
            output[channel * 256 + index] = value;
            effective = effective || value != index;
        }
    }
}

}  // namespace gpu
