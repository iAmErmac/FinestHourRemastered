#pragma once

#include "gpu/vulkan/buffer.h"
#include "gpu/vulkan/context.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gpu::vulkan {

// A persistently mapped suballocator whose blocks remain live until the owner has waited for its batch.
class UploadAllocator {
public:
    bool Init(const Context& context, VkDeviceSize minimum_block_size, std::string& error);
    bool Allocate(VkDeviceSize size, VkDeviceSize alignment, Buffer*& buffer, VkDeviceSize& offset,
                  std::string& error);
    void Reset();

    size_t BlockCount() const { return blocks_.size(); }
    size_t BlockIndex(const Buffer* buffer) const;
    VkDeviceSize BlockUsed(size_t index) const;
    VkDeviceSize BlockCapacity(size_t index) const;

private:
    struct Block {
        std::unique_ptr<Buffer> buffer;
        VkDeviceSize used = 0;
    };

    const Context* context_ = nullptr;
    VkDeviceSize minimum_block_size_ = 0;
    std::vector<Block> blocks_;
};

}  // namespace gpu::vulkan
