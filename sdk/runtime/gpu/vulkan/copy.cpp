#include "gpu/vulkan/copy.h"

#include "gpu/vulkan/context.h"

#include <algorithm>

namespace gpu::vulkan {

namespace {

uint32_t ScaleCoordinate(int64_t value, uint32_t host, uint32_t guest) {
    if (value <= 0 || !host || !guest) return 0;
    return std::min(host, uint32_t(uint64_t(value) * host / guest));
}

}  // namespace

bool BuildImageCopyRegions(uint32_t source_guest_width, uint32_t source_guest_height, uint32_t source_host_width,
                           uint32_t source_host_height, uint32_t destination_guest_width,
                           uint32_t destination_guest_height, uint32_t destination_host_width,
                           uint32_t destination_host_height, uint32_t count, const CopyRect* source_rects,
                           const CopyPoint* destination_points, std::vector<ImageCopyRegion>& regions) {
    regions.clear();
    if (!source_guest_width || !source_guest_height || !source_host_width || !source_host_height ||
        !destination_guest_width || !destination_guest_height || !destination_host_width || !destination_host_height)
        return false;
    const bool listed = count && source_rects;
    const uint32_t region_count = listed ? count : 1;
    regions.reserve(region_count);
    for (uint32_t i = 0; i < region_count; ++i) {
        int64_t x0 = 0, y0 = 0, x1 = std::min(source_guest_width, destination_guest_width);
        int64_t y1 = std::min(source_guest_height, destination_guest_height), dx = 0, dy = 0;
        if (listed) {
            const CopyRect& rect = source_rects[i];
            x0 = rect.left;
            y0 = rect.top;
            x1 = rect.right;
            y1 = rect.bottom;
            dx = x0;
            dy = y0;
        }
        if (destination_points) {
            dx = destination_points[i].x;
            dy = destination_points[i].y;
        }
        if (x1 <= x0 || y1 <= y0) continue;

        const int64_t source_left = std::max<int64_t>(x0, 0);
        const int64_t source_top = std::max<int64_t>(y0, 0);
        const int64_t source_right = std::min<int64_t>(x1, source_guest_width);
        const int64_t source_bottom = std::min<int64_t>(y1, source_guest_height);
        if (source_right <= source_left || source_bottom <= source_top) continue;

        int64_t destination_left = dx + source_left - x0;
        int64_t destination_top = dy + source_top - y0;
        int64_t clipped_left = source_left;
        int64_t clipped_top = source_top;
        if (destination_left < 0) {
            clipped_left += -destination_left;
            destination_left = 0;
        }
        if (destination_top < 0) {
            clipped_top += -destination_top;
            destination_top = 0;
        }
        int64_t copy_width = source_right - clipped_left;
        int64_t copy_height = source_bottom - clipped_top;
        copy_width = std::min<int64_t>(copy_width, int64_t(destination_guest_width) - destination_left);
        copy_height = std::min<int64_t>(copy_height, int64_t(destination_guest_height) - destination_top);
        if (copy_width <= 0 || copy_height <= 0) continue;

        const uint32_t source_x0 = ScaleCoordinate(clipped_left, source_host_width, source_guest_width);
        const uint32_t source_y0 = ScaleCoordinate(clipped_top, source_host_height, source_guest_height);
        const uint32_t source_x1 = ScaleCoordinate(clipped_left + copy_width, source_host_width, source_guest_width);
        const uint32_t source_y1 = ScaleCoordinate(clipped_top + copy_height, source_host_height, source_guest_height);
        const uint32_t destination_x0 =
            ScaleCoordinate(destination_left, destination_host_width, destination_guest_width);
        const uint32_t destination_y0 =
            ScaleCoordinate(destination_top, destination_host_height, destination_guest_height);
        const uint32_t destination_x1 =
            ScaleCoordinate(destination_left + copy_width, destination_host_width, destination_guest_width);
        const uint32_t destination_y1 =
            ScaleCoordinate(destination_top + copy_height, destination_host_height, destination_guest_height);
        if (source_x1 <= source_x0 || source_y1 <= source_y0 || destination_x1 <= destination_x0 ||
            destination_y1 <= destination_y0)
            continue;
        ImageCopyRegion region{};
        region.src_offset = {int32_t(source_x0), int32_t(source_y0), 0};
        region.dst_offset = {int32_t(destination_x0), int32_t(destination_y0), 0};
        region.src_extent = {source_x1 - source_x0, source_y1 - source_y0, 1};
        region.dst_extent = {destination_x1 - destination_x0, destination_y1 - destination_y0, 1};
        regions.push_back(region);
    }
    return true;
}

bool RecordImageTransfer(const Context& context, VkCommandBuffer command, VkImage source, VkImageLayout source_layout,
                         VkImage destination, VkImageLayout destination_layout, VkFormat format,
                         const std::vector<ImageCopyRegion>& regions, std::string& error, VkImageAspectFlags aspect) {
    if (regions.empty()) return true;
    auto copy = reinterpret_cast<PFN_vkCmdCopyImage>(context.DeviceProc("vkCmdCopyImage"));
    auto blit = reinterpret_cast<PFN_vkCmdBlitImage>(context.DeviceProc("vkCmdBlitImage"));
    bool has_copy = false;
    bool has_blit = false;
    uint32_t valid_regions = 0;
    for (const ImageCopyRegion& region : regions) {
        if (!region.src_extent.width || !region.src_extent.height || !region.dst_extent.width ||
            !region.dst_extent.height)
            continue;
        const bool resize = region.src_extent.width != region.dst_extent.width ||
                            region.src_extent.height != region.dst_extent.height ||
                            region.src_extent.depth != region.dst_extent.depth;
        ++valid_regions;
        has_copy = has_copy || !resize;
        has_blit = has_blit || resize;
    }
    if (has_copy && !copy) {
        error = "vkCmdCopyImage unavailable";
        return false;
    }
    VkFilter filter = VK_FILTER_NEAREST;
    if (has_blit) {
        if (format == VK_FORMAT_UNDEFINED) {
            error = "image blit format is unspecified";
            return false;
        }
        auto get_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceFormatProperties>(
            context.InstanceProc("vkGetPhysicalDeviceFormatProperties"));
        if (!get_properties || !blit) {
            error = "Vulkan image blit entry points unavailable";
            return false;
        }
        VkFormatProperties properties{};
        get_properties(context.PhysicalDevice(), format, &properties);
        constexpr VkFormatFeatureFlags required = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
        if ((properties.optimalTilingFeatures & required) != required) {
            error = "Vulkan image format lacks nearest-blit support";
            return false;
        }
        filter = (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)
                     ? VK_FILTER_LINEAR
                     : VK_FILTER_NEAREST;
    }
    auto barrier = reinterpret_cast<PFN_vkCmdPipelineBarrier>(context.DeviceProc("vkCmdPipelineBarrier"));
    if (valid_regions > 1 && !barrier) {
        error = "vkCmdPipelineBarrier unavailable for ordered image regions";
        return false;
    }
    bool first = true;
    for (const ImageCopyRegion& region : regions) {
        if (!region.src_extent.width || !region.src_extent.height || !region.dst_extent.width ||
            !region.dst_extent.height)
            continue;
        const bool resize = region.src_extent.width != region.dst_extent.width ||
                            region.src_extent.height != region.dst_extent.height ||
                            region.src_extent.depth != region.dst_extent.depth;
        if (!first) {
            VkImageMemoryBarrier dependency{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            dependency.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            dependency.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
            dependency.oldLayout = destination_layout;
            dependency.newLayout = destination_layout;
            dependency.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            dependency.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            dependency.image = destination;
            dependency.subresourceRange = {aspect, 0, 1, 0, 1};
            barrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                    nullptr, 1, &dependency);
        }
        if (!resize) {
            VkImageCopy copy_region{};
            copy_region.srcSubresource = {aspect, 0, 0, 1};
            copy_region.dstSubresource = {aspect, 0, 0, 1};
            copy_region.srcOffset = region.src_offset;
            copy_region.dstOffset = region.dst_offset;
            copy_region.extent = region.src_extent;
            copy(command, source, source_layout, destination, destination_layout, 1, &copy_region);
        } else {
            VkImageBlit blit_region{};
            blit_region.srcSubresource = {aspect, 0, 0, 1};
            blit_region.dstSubresource = {aspect, 0, 0, 1};
            blit_region.srcOffsets[0] = region.src_offset;
            blit_region.srcOffsets[1] = {region.src_offset.x + int32_t(region.src_extent.width),
                                         region.src_offset.y + int32_t(region.src_extent.height),
                                         region.src_offset.z + int32_t(region.src_extent.depth)};
            blit_region.dstOffsets[0] = region.dst_offset;
            blit_region.dstOffsets[1] = {region.dst_offset.x + int32_t(region.dst_extent.width),
                                         region.dst_offset.y + int32_t(region.dst_extent.height),
                                         region.dst_offset.z + int32_t(region.dst_extent.depth)};
            blit(command, source, source_layout, destination, destination_layout, 1, &blit_region, filter);
        }
        first = false;
    }
    return true;
}

}  // namespace gpu::vulkan
