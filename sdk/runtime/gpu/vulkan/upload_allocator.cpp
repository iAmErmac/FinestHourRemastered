#include "gpu/vulkan/upload_allocator.h"

#include <algorithm>
#include <limits>

namespace gpu::vulkan {
namespace {

VkDeviceSize Align(VkDeviceSize value, VkDeviceSize alignment) {
    const VkDeviceSize remainder = value % alignment;
    return remainder ? value + alignment - remainder : value;
}

}  // namespace

bool UploadAllocator::Init(const Context& context, VkDeviceSize minimum_block_size, std::string& error) {
    blocks_.clear();
    context_ = nullptr;
    minimum_block_size_ = 0;
    if (!context.Initialized() || !minimum_block_size) {
        error = "upload allocator requires an initialized context and nonzero block size";
        return false;
    }
    context_ = &context;
    minimum_block_size_ = minimum_block_size;
    error.clear();
    return true;
}

bool UploadAllocator::Allocate(VkDeviceSize size, VkDeviceSize alignment, Buffer*& buffer, VkDeviceSize& offset,
                               std::string& error) {
    buffer = nullptr;
    offset = 0;
    error.clear();
    if (!context_ || !context_->Initialized() || !size || !alignment || size > UINT32_MAX) {
        error = "invalid upload extent or alignment";
        return false;
    }
    // Search every block: a large request may skip a block's tail, and a later small request can use that tail.
    for (Block& block : blocks_) {
        const VkDeviceSize candidate = Align(block.used, alignment);
        if (candidate <= block.buffer->Size() && size <= block.buffer->Size() - candidate) {
            buffer = block.buffer.get();
            offset = candidate;
            block.used = candidate + size;
            return true;
        }
    }

    Block block;
    block.buffer = std::make_unique<Buffer>();
    const VkDeviceSize capacity = std::max(minimum_block_size_, size);
    if (!block.buffer->Init(*context_, capacity,
                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, error)) {
        return false;
    }
    buffer = block.buffer.get();
    block.used = size;
    blocks_.push_back(std::move(block));
    return true;
}

void UploadAllocator::Reset() {
    // The owner must call this only after the command batch using these buffers has completed.
    for (Block& block : blocks_) block.used = 0;
}

size_t UploadAllocator::BlockIndex(const Buffer* buffer) const {
    for (size_t i = 0; i < blocks_.size(); ++i)
        if (blocks_[i].buffer.get() == buffer) return i;
    return blocks_.size();
}

VkDeviceSize UploadAllocator::BlockUsed(size_t index) const {
    return index < blocks_.size() ? blocks_[index].used : 0;
}

VkDeviceSize UploadAllocator::BlockCapacity(size_t index) const {
    return index < blocks_.size() ? blocks_[index].buffer->Size() : 0;
}

}  // namespace gpu::vulkan
