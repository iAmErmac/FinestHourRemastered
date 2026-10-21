#pragma once

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <cstdint>

namespace gpu::vulkan {

// The XDK state uses zero for no culling and these two values for the directed modes.
inline constexpr uint32_t kGuestCullCw = 0x900u;
inline constexpr uint32_t kGuestCullCcw = 0x901u;

inline VkCullModeFlags RasterCullMode(uint32_t guest_cull, bool triangles) {
    return triangles && guest_cull ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
}

inline VkFrontFace RasterFrontFace(uint32_t guest_cull) {
    // The backend uses a negative-height viewport to keep the guest's top-left target origin. That
    // transform reverses framebuffer winding, so D3DCULL_CW's counter-clockwise target faces use
    // Vulkan's counter-clockwise enum here.
    return guest_cull == kGuestCullCw ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
}

}  // namespace gpu::vulkan
