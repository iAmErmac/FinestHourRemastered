#include "gpu/vulkan/buffer.h"

#include <algorithm>
#include <bit>
#include <cstring>

namespace gpu::vulkan {
namespace {

bool Result(VkResult result, const char* operation, std::string& error) {
    if (result == VK_SUCCESS) return true;
    error = std::string(operation) + " failed: VkResult " + std::to_string(static_cast<int>(result));
    return false;
}

}  // namespace

Buffer::~Buffer() { Reset(); }

void Buffer::Reset() {
    if (mapped_) unmap_memory_(device_, memory_);
    if (buffer_) destroy_buffer_(device_, buffer_, nullptr);
    if (memory_) free_memory_(device_, memory_, nullptr);
    device_ = VK_NULL_HANDLE;
    buffer_ = VK_NULL_HANDLE;
    memory_ = VK_NULL_HANDLE;
    mapped_ = nullptr;
    size_ = allocation_size_ = 0;
    atom_size_ = 1;
    memory_properties_ = 0;
    destroy_buffer_ = nullptr;
    free_memory_ = nullptr;
    unmap_memory_ = nullptr;
    flush_ranges_ = nullptr;
    invalidate_ranges_ = nullptr;
}

bool Buffer::Init(const Context& context, VkDeviceSize size, VkBufferUsageFlags usage,
                  VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred, std::string& error) {
    Reset();
    error.clear();
    if (!context.Initialized() || size == 0 || usage == 0) {
        error = "Vulkan buffer requires an initialized context, nonzero size and usage";
        return false;
    }
    device_ = context.Device();
    auto create_buffer = reinterpret_cast<PFN_vkCreateBuffer>(context.DeviceProc("vkCreateBuffer"));
    destroy_buffer_ = reinterpret_cast<PFN_vkDestroyBuffer>(context.DeviceProc("vkDestroyBuffer"));
    auto requirements = reinterpret_cast<PFN_vkGetBufferMemoryRequirements>(context.DeviceProc("vkGetBufferMemoryRequirements"));
    auto allocate = reinterpret_cast<PFN_vkAllocateMemory>(context.DeviceProc("vkAllocateMemory"));
    free_memory_ = reinterpret_cast<PFN_vkFreeMemory>(context.DeviceProc("vkFreeMemory"));
    auto bind = reinterpret_cast<PFN_vkBindBufferMemory>(context.DeviceProc("vkBindBufferMemory"));
    auto map = reinterpret_cast<PFN_vkMapMemory>(context.DeviceProc("vkMapMemory"));
    unmap_memory_ = reinterpret_cast<PFN_vkUnmapMemory>(context.DeviceProc("vkUnmapMemory"));
    flush_ranges_ = reinterpret_cast<PFN_vkFlushMappedMemoryRanges>(context.DeviceProc("vkFlushMappedMemoryRanges"));
    invalidate_ranges_ = reinterpret_cast<PFN_vkInvalidateMappedMemoryRanges>(context.DeviceProc("vkInvalidateMappedMemoryRanges"));
    auto properties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(context.InstanceProc("vkGetPhysicalDeviceMemoryProperties"));
    if (!create_buffer || !destroy_buffer_ || !requirements || !allocate || !free_memory_ || !bind || !map ||
        !unmap_memory_ || !flush_ranges_ || !invalidate_ranges_ || !properties) {
        error = "Vulkan buffer allocation entry points are unavailable";
        Reset();
        return false;
    }
    VkBufferCreateInfo create{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    create.size = size;
    create.usage = usage;
    create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (!Result(create_buffer(device_, &create, nullptr, &buffer_), "vkCreateBuffer", error)) {
        Reset();
        return false;
    }
    VkMemoryRequirements needed{};
    requirements(device_, buffer_, &needed);
    VkPhysicalDeviceMemoryProperties available{};
    properties(context.PhysicalDevice(), &available);
    uint32_t selected = UINT32_MAX;
    int best_score = -1;
    for (uint32_t i = 0; i < available.memoryTypeCount; ++i) {
        const auto flags = available.memoryTypes[i].propertyFlags;
        if (!(needed.memoryTypeBits & (1u << i)) || (flags & required) != required) continue;
        const int score = std::popcount(flags & preferred);
        if (score > best_score) {
            selected = i;
            best_score = score;
        }
    }
    if (selected == UINT32_MAX) {
        error = "No compatible Vulkan buffer memory type satisfies required properties";
        Reset();
        return false;
    }
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = needed.size;
    allocation.memoryTypeIndex = selected;
    if (!Result(allocate(device_, &allocation, nullptr, &memory_), "vkAllocateMemory", error) ||
        !Result(bind(device_, buffer_, memory_, 0), "vkBindBufferMemory", error)) {
        Reset();
        return false;
    }
    size_ = size;
    allocation_size_ = needed.size;
    atom_size_ = std::max<VkDeviceSize>(1, context.Properties().limits.nonCoherentAtomSize);
    memory_properties_ = available.memoryTypes[selected].propertyFlags;
    if (memory_properties_ & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        if (!Result(map(device_, memory_, 0, VK_WHOLE_SIZE, 0, &mapped_), "vkMapMemory", error)) {
            Reset();
            return false;
        }
    }
    return true;
}

bool Buffer::CheckRange(VkDeviceSize offset, size_t size, std::string& error) const {
    if (!mapped_) {
        error = "Vulkan buffer is not mapped host-visible memory";
        return false;
    }
    if (offset > size_ || size > size_ - offset) {
        error = "Vulkan buffer host access is outside its logical size";
        return false;
    }
    return true;
}

bool Buffer::MaintainRange(VkDeviceSize offset, size_t size, bool flush, std::string& error) {
    if (!size || (memory_properties_ & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) return true;
    const VkDeviceSize begin = offset - offset % atom_size_;
    const VkDeviceSize end = offset + size;
    const VkDeviceSize padding = (atom_size_ - end % atom_size_) % atom_size_;
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    range.memory = memory_;
    range.offset = begin;
    // Ending at the allocation boundary is valid even if its size is not atom-aligned.
    range.size = padding > allocation_size_ - end ? allocation_size_ - begin : end + padding - begin;
    const auto result = flush ? flush_ranges_(device_, 1, &range) : invalidate_ranges_(device_, 1, &range);
    return Result(result, flush ? "vkFlushMappedMemoryRanges" : "vkInvalidateMappedMemoryRanges", error);
}

bool Buffer::Write(VkDeviceSize offset, const void* data, size_t size, std::string& error) {
    error.clear();
    if (!CheckRange(offset, size, error)) return false;
    if (size && !data) {
        error = "Vulkan buffer write requires source data";
        return false;
    }
    if (size) std::memcpy(static_cast<std::byte*>(mapped_) + offset, data, size);
    return MaintainRange(offset, size, true, error);
}

bool Buffer::Read(VkDeviceSize offset, void* data, size_t size, std::string& error) {
    error.clear();
    if (!CheckRange(offset, size, error)) return false;
    if (size && !data) {
        error = "Vulkan buffer read requires destination storage";
        return false;
    }
    if (!MaintainRange(offset, size, false, error)) return false;
    if (size) std::memcpy(data, static_cast<std::byte*>(mapped_) + offset, size);
    return true;
}

}  // namespace gpu::vulkan
