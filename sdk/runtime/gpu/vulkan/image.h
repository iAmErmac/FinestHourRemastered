#pragma once

#include "gpu/vulkan/context.h"

namespace gpu::vulkan {

// Retire all GPU uses and destroy the view/image before resetting the parent context.
class Image {
public:
    Image() = default;
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    ~Image();

    bool Init(const Context& context, uint32_t width, uint32_t height, VkFormat format,
              VkImageUsageFlags usage, VkImageAspectFlags aspect, VkSampleCountFlagBits samples,
              std::string& error, uint32_t depth = 1, uint32_t layers = 1, uint32_t levels = 1,
              VkImageViewType view_type = VK_IMAGE_VIEW_TYPE_2D);
    void Reset();
    VkImage Handle() const { return image_; }
    VkImageView View() const { return view_; }
    // Depth/stencil images keep an attachment view and a depth-only sampled view.
    VkImageView SampledView() const { return sampled_view_ ? sampled_view_ : view_; }
    uint32_t MipLevels() const { return mip_levels_; }
    uint32_t ArrayLayers() const { return array_layers_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkImage image_ = VK_NULL_HANDLE;
    VkImageView view_ = VK_NULL_HANDLE;
    VkImageView sampled_view_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    uint32_t mip_levels_ = 1;
    uint32_t array_layers_ = 1;
    PFN_vkDestroyImage destroy_image_ = nullptr;
    PFN_vkDestroyImageView destroy_view_ = nullptr;
    PFN_vkFreeMemory free_memory_ = nullptr;
};

}  // namespace gpu::vulkan
