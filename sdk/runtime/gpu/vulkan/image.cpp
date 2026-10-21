#include "gpu/vulkan/image.h"

#include <bit>

namespace gpu::vulkan {
namespace {

bool Result(VkResult result, const char* operation, std::string& error) {
    if (result == VK_SUCCESS) return true;
    error = std::string(operation) + " failed: VkResult " + std::to_string(static_cast<int>(result));
    return false;
}

}  // namespace

Image::~Image() { Reset(); }

void Image::Reset() {
    if (sampled_view_ && sampled_view_ != view_) destroy_view_(device_, sampled_view_, nullptr);
    if (view_) destroy_view_(device_, view_, nullptr);
    if (image_) destroy_image_(device_, image_, nullptr);
    if (memory_) free_memory_(device_, memory_, nullptr);
    device_ = VK_NULL_HANDLE;
    view_ = VK_NULL_HANDLE;
    sampled_view_ = VK_NULL_HANDLE;
    image_ = VK_NULL_HANDLE;
    memory_ = VK_NULL_HANDLE;
    mip_levels_ = 1;
    array_layers_ = 1;
    destroy_view_ = nullptr;
    destroy_image_ = nullptr;
    free_memory_ = nullptr;
}

bool Image::Init(const Context& context, uint32_t width, uint32_t height, VkFormat format,
                 VkImageUsageFlags usage, VkImageAspectFlags aspect, VkSampleCountFlagBits samples,
                 std::string& error, uint32_t depth, uint32_t layers, uint32_t levels,
                 VkImageViewType view_type) {
    Reset();
    error.clear();
    if (!context.Initialized() || !width || !height || !usage || !aspect ||
        !depth || !layers || !levels || !std::has_single_bit(static_cast<uint32_t>(samples)) ||
        format == VK_FORMAT_UNDEFINED ||
        (view_type == VK_IMAGE_VIEW_TYPE_CUBE && (depth != 1 || layers != 6 || width != height)) ||
        (view_type == VK_IMAGE_VIEW_TYPE_3D && layers != 1) ||
        (view_type != VK_IMAGE_VIEW_TYPE_2D && view_type != VK_IMAGE_VIEW_TYPE_2D_ARRAY &&
         view_type != VK_IMAGE_VIEW_TYPE_CUBE && view_type != VK_IMAGE_VIEW_TYPE_3D)) {
        error = "Vulkan image requires a context, valid extent, format, usage, aspect, sample count and view shape";
        return false;
    }
    const VkImageType image_type = view_type == VK_IMAGE_VIEW_TYPE_3D ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
    const uint32_t array_layers = image_type == VK_IMAGE_TYPE_3D ? 1 : layers;
    const VkImageCreateFlags create_flags = view_type == VK_IMAGE_VIEW_TYPE_CUBE ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
    device_ = context.Device();
    auto create_image = reinterpret_cast<PFN_vkCreateImage>(context.DeviceProc("vkCreateImage"));
    destroy_image_ = reinterpret_cast<PFN_vkDestroyImage>(context.DeviceProc("vkDestroyImage"));
    auto requirements = reinterpret_cast<PFN_vkGetImageMemoryRequirements>(context.DeviceProc("vkGetImageMemoryRequirements"));
    auto allocate = reinterpret_cast<PFN_vkAllocateMemory>(context.DeviceProc("vkAllocateMemory"));
    free_memory_ = reinterpret_cast<PFN_vkFreeMemory>(context.DeviceProc("vkFreeMemory"));
    auto bind = reinterpret_cast<PFN_vkBindImageMemory>(context.DeviceProc("vkBindImageMemory"));
    auto create_view = reinterpret_cast<PFN_vkCreateImageView>(context.DeviceProc("vkCreateImageView"));
    destroy_view_ = reinterpret_cast<PFN_vkDestroyImageView>(context.DeviceProc("vkDestroyImageView"));
    auto properties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(context.InstanceProc("vkGetPhysicalDeviceMemoryProperties"));
    auto supported = reinterpret_cast<PFN_vkGetPhysicalDeviceImageFormatProperties>(context.InstanceProc("vkGetPhysicalDeviceImageFormatProperties"));
    if (!create_image || !destroy_image_ || !requirements || !allocate || !free_memory_ || !bind || !create_view ||
        !destroy_view_ || !properties || !supported) {
        error = "Vulkan image allocation entry points are unavailable";
        Reset();
        return false;
    }
    VkImageFormatProperties limits{};
    if (!Result(supported(context.PhysicalDevice(), format, image_type, VK_IMAGE_TILING_OPTIMAL,
                          usage, create_flags, &limits), "vkGetPhysicalDeviceImageFormatProperties", error)) {
        Reset();
        return false;
    }
    if (width > limits.maxExtent.width || height > limits.maxExtent.height || depth > limits.maxExtent.depth ||
        array_layers > limits.maxArrayLayers || levels > limits.maxMipLevels || !(limits.sampleCounts & samples)) {
        error = "Vulkan image extent or samples exceed format/usage capabilities";
        Reset();
        return false;
    }
    VkImageCreateInfo create{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    create.flags = create_flags;
    create.imageType = image_type;
    create.format = format;
    create.extent = {width, height, image_type == VK_IMAGE_TYPE_3D ? depth : 1};
    create.mipLevels = levels;
    create.arrayLayers = array_layers;
    create.samples = samples;
    create.tiling = VK_IMAGE_TILING_OPTIMAL;
    create.usage = usage;
    create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    create.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (!Result(create_image(device_, &create, nullptr, &image_), "vkCreateImage", error)) {
        Reset();
        return false;
    }
    VkMemoryRequirements needed{};
    requirements(device_, image_, &needed);
    VkPhysicalDeviceMemoryProperties available{};
    properties(context.PhysicalDevice(), &available);
    uint32_t selected = UINT32_MAX;
    for (uint32_t i = 0; i < available.memoryTypeCount; ++i) {
        if ((needed.memoryTypeBits & (1u << i)) &&
            (available.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            selected = i;
            break;
        }
    }
    if (selected == UINT32_MAX) {
        error = "No compatible device-local Vulkan image memory type";
        Reset();
        return false;
    }
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = needed.size;
    allocation.memoryTypeIndex = selected;
    if (!Result(allocate(device_, &allocation, nullptr, &memory_), "vkAllocateMemory", error) ||
        !Result(bind(device_, image_, memory_, 0), "vkBindImageMemory", error)) {
        Reset();
        return false;
    }
    const VkImageUsageFlags view_usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                        VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
    if (usage & view_usage) {
        VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view.image = image_;
        view.viewType = view_type;
        view.format = format;
        view.subresourceRange = {aspect, 0, levels, 0, array_layers};
        if (!Result(create_view(device_, &view, nullptr, &view_), "vkCreateImageView", error)) {
            Reset();
            return false;
        }
        if ((aspect & VK_IMAGE_ASPECT_DEPTH_BIT) && (aspect & VK_IMAGE_ASPECT_STENCIL_BIT) &&
            (usage & VK_IMAGE_USAGE_SAMPLED_BIT)) {
            VkImageViewCreateInfo sampled = view;
            sampled.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            if (!Result(create_view(device_, &sampled, nullptr, &sampled_view_), "vkCreateImageView depth", error)) {
                Reset();
                return false;
            }
        }
    }
    mip_levels_ = levels;
    array_layers_ = array_layers;
    return true;
}

}  // namespace gpu::vulkan
