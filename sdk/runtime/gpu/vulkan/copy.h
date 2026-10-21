#pragma once

#include <string>
#include <vector>

#include <vulkan/vulkan.h>

namespace gpu::vulkan {

class Context;

struct ImageCopyRegion {
    VkOffset3D src_offset{};
    VkOffset3D dst_offset{};
    VkExtent3D src_extent{};
    VkExtent3D dst_extent{};
};

struct CopyRect {
    int32_t left = 0;
    int32_t top = 0;
    int32_t right = 0;
    int32_t bottom = 0;
};

struct CopyPoint {
    int32_t x = 0;
    int32_t y = 0;
};

bool BuildImageCopyRegions(uint32_t source_guest_width, uint32_t source_guest_height, uint32_t source_host_width,
                           uint32_t source_host_height, uint32_t destination_guest_width,
                           uint32_t destination_guest_height, uint32_t destination_host_width,
                           uint32_t destination_host_height, uint32_t count, const CopyRect* source_rects,
                           const CopyPoint* destination_points, std::vector<ImageCopyRegion>& regions);

bool RecordImageTransfer(const Context& context, VkCommandBuffer command, VkImage source, VkImageLayout source_layout,
                         VkImage destination, VkImageLayout destination_layout, VkFormat format,
                         const std::vector<ImageCopyRegion>& regions, std::string& error,
                         VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);

}  // namespace gpu::vulkan
