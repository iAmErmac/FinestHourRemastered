#include "gpu/vulkan/commands.h"

#include "core/log.h"

#include <cstdlib>

namespace gpu::vulkan {
namespace {

bool Result(VkResult result, const char* operation, std::string& error) {
    if (result == VK_SUCCESS) return true;
    error = std::string(operation) + " failed: VkResult " + std::to_string(static_cast<int>(result));
    return false;
}

}  // namespace

CommandBatch::~CommandBatch() { Reset(); }

void CommandBatch::Reset() {
    // Final cleanup cannot leave a live command pool behind its owner.
    if (pending_) {
        const VkFence completion = external_pending_ ? external_fence_ : fence_;
        if (!completion) {
            LOGE("vulkan: pending command batch has no completion fence");
            std::abort();
        }
        const VkResult result = wait_(device_, 1, &completion, VK_TRUE, UINT64_MAX);
        if (result != VK_SUCCESS && result != VK_ERROR_DEVICE_LOST) {
            LOGE("vulkan: command completion failed during final cleanup (%d)", int(result));
            std::abort();
        }
    }
    if (fence_) destroy_fence_(device_, fence_, nullptr);
    if (pool_) destroy_pool_(device_, pool_, nullptr);
    device_ = VK_NULL_HANDLE;
    queue_ = VK_NULL_HANDLE;
    pool_ = VK_NULL_HANDLE;
    command_ = VK_NULL_HANDLE;
    fence_ = VK_NULL_HANDLE;
    recording_ = pending_ = false;
    ready_for_external_submit_ = false;
    external_pending_ = false;
    external_fence_ = VK_NULL_HANDLE;
    submission_serial_ = 0;
    completed_serial_ = 0;
    destroy_pool_ = nullptr;
    reset_pool_ = nullptr;
    begin_ = nullptr;
    end_ = nullptr;
    destroy_fence_ = nullptr;
    reset_fences_ = nullptr;
    get_fence_status_ = nullptr;
    submit_ = nullptr;
    wait_ = nullptr;
}

bool CommandBatch::Init(const Context& context, std::string& error) {
    Reset();
    error.clear();
    if (pending_) {
        error = "Vulkan command batch could not finish its previous submission";
        return false;
    }
    if (!context.Initialized()) {
        error = "Vulkan command batch requires an initialized context";
        return false;
    }
    device_ = context.Device();
    queue_ = context.GraphicsQueue();
    auto create_pool = reinterpret_cast<PFN_vkCreateCommandPool>(context.DeviceProc("vkCreateCommandPool"));
    destroy_pool_ = reinterpret_cast<PFN_vkDestroyCommandPool>(context.DeviceProc("vkDestroyCommandPool"));
    reset_pool_ = reinterpret_cast<PFN_vkResetCommandPool>(context.DeviceProc("vkResetCommandPool"));
    auto allocate = reinterpret_cast<PFN_vkAllocateCommandBuffers>(context.DeviceProc("vkAllocateCommandBuffers"));
    begin_ = reinterpret_cast<PFN_vkBeginCommandBuffer>(context.DeviceProc("vkBeginCommandBuffer"));
    end_ = reinterpret_cast<PFN_vkEndCommandBuffer>(context.DeviceProc("vkEndCommandBuffer"));
    auto create_fence = reinterpret_cast<PFN_vkCreateFence>(context.DeviceProc("vkCreateFence"));
    destroy_fence_ = reinterpret_cast<PFN_vkDestroyFence>(context.DeviceProc("vkDestroyFence"));
    reset_fences_ = reinterpret_cast<PFN_vkResetFences>(context.DeviceProc("vkResetFences"));
    get_fence_status_ = reinterpret_cast<PFN_vkGetFenceStatus>(context.DeviceProc("vkGetFenceStatus"));
    submit_ = reinterpret_cast<PFN_vkQueueSubmit2>(context.DeviceProc("vkQueueSubmit2"));
    wait_ = reinterpret_cast<PFN_vkWaitForFences>(context.DeviceProc("vkWaitForFences"));
    if (!create_pool || !destroy_pool_ || !reset_pool_ || !allocate || !begin_ || !end_ || !create_fence ||
        !destroy_fence_ || !reset_fences_ || !get_fence_status_ || !submit_ || !wait_) {
        error = "Vulkan command batch entry points are unavailable";
        Reset();
        return false;
    }
    VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool.queueFamilyIndex = context.GraphicsQueueFamily();
    if (!Result(create_pool(device_, &pool, nullptr, &pool_), "vkCreateCommandPool", error)) {
        Reset();
        return false;
    }
    VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation.commandPool = pool_;
    allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation.commandBufferCount = 1;
    VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (!Result(allocate(device_, &allocation, &command_), "vkAllocateCommandBuffers", error) ||
        !Result(create_fence(device_, &fence, nullptr, &fence_), "vkCreateFence", error)) {
        Reset();
        return false;
    }
    return true;
}

bool CommandBatch::Begin(VkCommandBuffer& command, std::string& error) {
    command = VK_NULL_HANDLE;
    error.clear();
    if (!command_ || recording_ || pending_) {
        error = "Vulkan command batch must be initialized and completed before recording";
        return false;
    }
    if (!Result(reset_pool_(device_, pool_, 0), "vkResetCommandPool", error)) return false;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (!Result(begin_(command_, &begin), "vkBeginCommandBuffer", error)) return false;
    recording_ = true;
    ready_for_external_submit_ = false;
    command = command_;
    return true;
}

bool CommandBatch::Submit(std::string& error) {
    error.clear();
    if (!recording_ || pending_) {
        error = "Vulkan command batch is not recording or is already pending";
        return false;
    }
    recording_ = false;
    if (!Result(end_(command_), "vkEndCommandBuffer", error) ||
        !Result(reset_fences_(device_, 1, &fence_), "vkResetFences", error)) return false;
    VkCommandBufferSubmitInfo command{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    command.commandBuffer = command_;
    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &command;
    if (!Result(submit_(queue_, 1, &submit, fence_), "vkQueueSubmit2", error)) return false;
    pending_ = true;
    submission_serial_.fetch_add(1, std::memory_order_release);
    return true;
}

bool CommandBatch::EndForExternalSubmit(std::string& error) {
    error.clear();
    if (!recording_ || pending_) {
        error = "Vulkan command batch is not recording for an external submit";
        return false;
    }
    recording_ = false;
    if (!Result(end_(command_), "vkEndCommandBuffer", error)) return false;
    ready_for_external_submit_ = true;
    return true;
}

bool CommandBatch::MarkExternalSubmitted(VkFence completion_fence, std::string& error) {
    error.clear();
    if (!device_ || !ready_for_external_submit_ || recording_ || pending_ || !completion_fence) {
        error = "Vulkan external command submission requires an ended batch and completion fence";
        return false;
    }
    ready_for_external_submit_ = false;
    external_pending_ = pending_ = true;
    external_fence_ = completion_fence;
    submission_serial_.fetch_add(1, std::memory_order_release);
    return true;
}

VkResult CommandBatch::CompletionStatus() const {
    if (!pending_) return VK_SUCCESS;
    const VkFence completion = external_pending_ ? external_fence_ : fence_;
    if (!completion || !get_fence_status_) return VK_ERROR_INITIALIZATION_FAILED;
    return get_fence_status_(device_, completion);
}

bool CommandBatch::IsComplete() const {
    return CompletionStatus() == VK_SUCCESS;
}

uint64_t CommandBatch::CompletedSerial() const {
    return CompletionStatus() == VK_SUCCESS ? submission_serial_.load(std::memory_order_acquire)
                                             : completed_serial_.load(std::memory_order_acquire);
}

bool CommandBatch::Wait(uint64_t timeout_ns, std::string& error) {
    error.clear();
    if (!device_ || recording_) {
        error = "Vulkan command batch must be initialized and submitted before waiting";
        return false;
    }
    if (!pending_) return true;
    if (external_pending_) {
        const VkResult result = wait_(device_, 1, &external_fence_, VK_TRUE, timeout_ns);
        if (!Result(result, "external Vulkan command completion", error)) return false;
        pending_ = external_pending_ = false;
        external_fence_ = VK_NULL_HANDLE;
        completed_serial_.store(submission_serial_.load(std::memory_order_acquire), std::memory_order_release);
        return true;
    }
    if (!Result(wait_(device_, 1, &fence_, VK_TRUE, timeout_ns), "vkWaitForFences", error)) return false;
    pending_ = false;
    completed_serial_.store(submission_serial_.load(std::memory_order_acquire), std::memory_order_release);
    return true;
}

}  // namespace gpu::vulkan
