#pragma once

#include "gpu/vulkan/context.h"

#include <atomic>

namespace gpu::vulkan {

// One render-thread-owned batch. Referenced resources must remain alive until Wait succeeds.
class CommandBatch {
public:
    CommandBatch() = default;
    CommandBatch(const CommandBatch&) = delete;
    CommandBatch& operator=(const CommandBatch&) = delete;
    ~CommandBatch();

    bool Init(const Context& context, std::string& error);
    bool Begin(VkCommandBuffer& command, std::string& error);
    bool Submit(std::string& error);
    // Swapchain owns the acquire/present semaphore submit. These methods let the
    // same batch own the command pool while that component owns the submission.
    bool EndForExternalSubmit(std::string& error);
    // The submitting owner must keep this fence alive and unchanged until Wait
    // retires the batch, including before swapchain recreation or shutdown.
    bool MarkExternalSubmitted(VkFence completion_fence, std::string& error);
    // A timeout retains the batch and its referenced resources for a later wait.
    bool Wait(uint64_t timeout_ns, std::string& error);
    // Submission serials let bounded asynchronous users retain resources until the exact batch completes.
    uint64_t SubmissionSerial() const { return submission_serial_.load(std::memory_order_acquire); }
    uint64_t CompletedSerial() const;
    // Returns VK_NOT_READY while pending, or the actual fence status for diagnostics/error handling.
    VkResult CompletionStatus() const;
    bool IsComplete() const;
    void Reset();
    bool Pending() const { return pending_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    bool recording_ = false, pending_ = false, external_pending_ = false;
    bool ready_for_external_submit_ = false;
    VkFence external_fence_ = VK_NULL_HANDLE;
    PFN_vkDestroyCommandPool destroy_pool_ = nullptr;
    PFN_vkResetCommandPool reset_pool_ = nullptr;
    PFN_vkBeginCommandBuffer begin_ = nullptr;
    PFN_vkEndCommandBuffer end_ = nullptr;
    PFN_vkDestroyFence destroy_fence_ = nullptr;
    PFN_vkResetFences reset_fences_ = nullptr;
    PFN_vkGetFenceStatus get_fence_status_ = nullptr;
    PFN_vkQueueSubmit2 submit_ = nullptr;
    PFN_vkWaitForFences wait_ = nullptr;
    std::atomic<uint64_t> submission_serial_{0};
    std::atomic<uint64_t> completed_serial_{0};
};

}  // namespace gpu::vulkan
