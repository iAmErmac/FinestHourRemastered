#pragma once

#include "gpu/vulkan/context.h"

#include <cstddef>
#include <string>

namespace gpu::vulkan {

// The context and all GPU uses must outlive the allocation; retire it after completion.
class Buffer {
public:
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer();

    bool Init(const Context& context, VkDeviceSize size, VkBufferUsageFlags usage,
              VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred, std::string& error);
    void Reset();
    bool Write(VkDeviceSize offset, const void* data, size_t size, std::string& error);
    bool Read(VkDeviceSize offset, void* data, size_t size, std::string& error);

    VkBuffer Handle() const { return buffer_; }
    VkDeviceSize Size() const { return size_; }
    VkMemoryPropertyFlags MemoryProperties() const { return memory_properties_; }

private:
    bool CheckRange(VkDeviceSize offset, size_t size, std::string& error) const;
    bool MaintainRange(VkDeviceSize offset, size_t size, bool flush, std::string& error);

    VkDevice device_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0, allocation_size_ = 0, atom_size_ = 1;
    VkMemoryPropertyFlags memory_properties_ = 0;
    void* mapped_ = nullptr;
    PFN_vkDestroyBuffer destroy_buffer_ = nullptr;
    PFN_vkFreeMemory free_memory_ = nullptr;
    PFN_vkUnmapMemory unmap_memory_ = nullptr;
    PFN_vkFlushMappedMemoryRanges flush_ranges_ = nullptr;
    PFN_vkInvalidateMappedMemoryRanges invalidate_ranges_ = nullptr;
};

}  // namespace gpu::vulkan
