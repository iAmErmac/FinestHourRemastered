#pragma once

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

namespace gpu::vulkan {

class Context;
class Win32Surface;

enum class PresentPreference {
    Fifo,
    Mailbox,
    Immediate,
};

enum class SwapchainInitStatus {
    Ready,
    Suspended,
    CapabilityLimited,
    SurfaceLost,
    DeviceLost,
    Failed,
};

enum class AcquireStatus {
    Acquired,
    RecoveryRequired,
    Suspended,
    NotReady,
    Timeout,
    RecreateRequired,
    SurfaceLost,
    DeviceLost,
    Failed,
};

enum class PresentStatus {
    Presented,
    RecreateRequired,
    SurfaceLost,
    DeviceLost,
    Failed,
};

enum class SubmitWaitPolicy {
    AllCommands,
    ColorAttachmentOutput,
};

struct AcquireResult {
    AcquireStatus status = AcquireStatus::Failed;
    const void* owner = nullptr;
    uint64_t generation = 0;
    uint32_t image_index = UINT32_MAX;
    uint32_t acquire_slot = UINT32_MAX;
    VkSemaphore acquire_semaphore = VK_NULL_HANDLE;
    uint64_t acquisition_serial = 0;
    bool suboptimal = false;

    bool Succeeded() const { return status == AcquireStatus::Acquired; }
    bool NeedsRecovery() const { return status == AcquireStatus::RecoveryRequired; }
};

class Swapchain {
public:
    Swapchain() = default;
    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;
    ~Swapchain();

    // The Context and Win32Surface must outlive this object. The supplied extent is in physical pixels.
    SwapchainInitStatus Init(const Context& context, const Win32Surface& surface, VkExtent2D extent,
                             PresentPreference preference, std::string& error);
    SwapchainInitStatus Recreate(VkExtent2D extent, PresentPreference preference, std::string& error);

    // Shutdown waits for the actual present fences before destroying image views, semaphores and the chain.
    // A failed shutdown leaves resources intact so the caller can retry before destroying the Context.
    bool Shutdown(std::string& error);
    void Reset();

    AcquireResult Acquire(uint64_t timeout_ns, std::string& error);
    // Records the caller's complete rendering command buffer. The component owns the acquire wait, render-complete
    // signal and render fence, so abort/retry paths never guess whether the acquire semaphore was consumed.
    bool Submit(const AcquireResult& result, VkCommandBuffer command_buffer, std::string& error,
                SubmitWaitPolicy wait_policy = SubmitWaitPolicy::AllCommands);
    VkFence RenderFence(const AcquireResult& result) const;
    // Returns the reusable render fence for an acquired image before Submit resets it. Acquire has already
    // waited any previous owner of this image, so callers must acknowledge that owner before submitting again.
    VkFence RenderFenceForImage(uint32_t image_index) const;
    // Waits for all submitted rendering command buffers before their command pools or resources are reused.
    bool WaitForRendering(uint64_t timeout_ns, std::string& error);
    // Present consumes the image's render-complete semaphore after Submit succeeds. Its present fence is separate
    // from the render fence and governs swapchain image-view/semaphore retirement.
    PresentStatus Present(const AcquireResult& result, std::string& error);
    // Releases an acquired image after a failed/aborted frame. It consumes exactly the semaphore that is still
    // signaled: the acquire semaphore before Submit, or the render-complete semaphore after Submit.
    bool Abandon(const AcquireResult& result, std::string& error);

    VkImage Image(const AcquireResult& result) const;
    VkImageView View(const AcquireResult& result) const;
    VkSemaphore PresentSemaphore(const AcquireResult& result) const;

    bool Ready() const { return swapchain_ != VK_NULL_HANDLE; }
    bool PresentFenceRetirement() const { return present_fence_retirement_; }
    VkFormat Format() const { return format_; }
    VkColorSpaceKHR ColorSpace() const { return color_space_; }
    VkExtent2D Extent() const { return extent_; }
    VkImageUsageFlags ImageUsage() const { return image_usage_; }
    VkSurfaceTransformFlagBitsKHR PreTransform() const { return pre_transform_; }
    VkCompositeAlphaFlagBitsKHR CompositeAlpha() const { return composite_alpha_; }
    VkPresentModeKHR PresentMode() const { return present_mode_; }
    uint32_t ImageCount() const { return static_cast<uint32_t>(images_.size()); }
    uint64_t Generation() const { return generation_; }
    uint64_t PresentedCount() const { return presented_count_; }

private:
    struct AcquireSlot {
        VkSemaphore semaphore = VK_NULL_HANDLE;
        uint32_t image_index = UINT32_MAX;
        bool in_flight = false;
    };

    bool LoadFunctions(std::string& error);
    SwapchainInitStatus Create(VkExtent2D requested_extent, PresentPreference preference, std::string& error);
    bool WaitForPresentCompletion(uint64_t timeout_ns, std::string& error);
    bool WaitForImagePresent(uint32_t image_index, uint64_t timeout_ns, std::string& error);
    bool FindAcquireSlot(uint64_t timeout_ns, uint32_t& slot, std::string& error);
    bool SubmitSemaphoreWait(VkSemaphore semaphore, uint64_t timeout_ns, uint64_t serial, std::string& error);
    bool AbandonInternal(const AcquireResult& result, uint64_t timeout_ns, std::string& error);
    bool WaitForRenderCompletion(uint32_t image_index, uint64_t timeout_ns, std::string& error);
    bool ValidateResult(const AcquireResult& result, std::string& error) const;
    bool ValidateAbandonResult(const AcquireResult& result, std::string& error) const;
    void MarkDeviceLost(VkResult result);
    void DestroyResources();
    void ClearOwner();
    static VkPresentModeKHR ChoosePresentMode(const std::vector<VkPresentModeKHR>& modes,
                                              PresentPreference preference);

    const Context* context_ = nullptr;
    const Win32Surface* surface_ = nullptr;
    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = VK_QUEUE_FAMILY_IGNORED;
    VkSurfaceKHR surface_handle_ = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkColorSpaceKHR color_space_ = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    VkExtent2D extent_{};
    VkImageUsageFlags image_usage_ = 0;
    VkSurfaceTransformFlagBitsKHR pre_transform_ = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    VkCompositeAlphaFlagBitsKHR composite_alpha_ = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    VkPresentModeKHR present_mode_ = VK_PRESENT_MODE_FIFO_KHR;
    uint64_t generation_ = 0;
    uint64_t presented_count_ = 0;
    uint32_t acquired_image_ = UINT32_MAX;
    uint32_t acquired_slot_ = UINT32_MAX;
    uint64_t acquisition_serial_ = 0;
    uint64_t acquired_serial_ = 0;
    VkFence pending_cleanup_fence_ = VK_NULL_HANDLE;
    uint64_t pending_cleanup_serial_ = 0;
    bool cleanup_wait_consumed_ = false;
    bool submission_submitted_ = false;
    bool device_lost_ = false;
    bool present_fence_retirement_ = false;

    std::vector<VkImage> images_;
    std::vector<VkImageView> views_;
    std::vector<VkSemaphore> present_semaphores_;
    std::vector<VkFence> present_fences_;
    std::vector<VkFence> render_fences_;
    std::vector<bool> present_pending_;
    std::vector<bool> present_fence_armed_;
    std::vector<bool> render_pending_;
    std::vector<AcquireSlot> acquire_slots_;

    PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR get_surface_capabilities_ = nullptr;
    PFN_vkGetPhysicalDeviceSurfaceFormatsKHR get_surface_formats_ = nullptr;
    PFN_vkGetPhysicalDeviceSurfacePresentModesKHR get_surface_modes_ = nullptr;
    PFN_vkCreateSwapchainKHR create_swapchain_ = nullptr;
    PFN_vkDestroySwapchainKHR destroy_swapchain_ = nullptr;
    PFN_vkGetSwapchainImagesKHR get_swapchain_images_ = nullptr;
    PFN_vkAcquireNextImageKHR acquire_next_image_ = nullptr;
    PFN_vkQueuePresentKHR queue_present_ = nullptr;
    PFN_vkCreateImageView create_image_view_ = nullptr;
    PFN_vkDestroyImageView destroy_image_view_ = nullptr;
    PFN_vkCreateSemaphore create_semaphore_ = nullptr;
    PFN_vkDestroySemaphore destroy_semaphore_ = nullptr;
    PFN_vkCreateFence create_fence_ = nullptr;
    PFN_vkDestroyFence destroy_fence_ = nullptr;
    PFN_vkGetFenceStatus get_fence_status_ = nullptr;
    PFN_vkWaitForFences wait_for_fences_ = nullptr;
    PFN_vkResetFences reset_fences_ = nullptr;
    PFN_vkQueueSubmit2 queue_submit2_ = nullptr;
    PFN_vkReleaseSwapchainImagesEXT release_swapchain_images_ = nullptr;
};

}  // namespace gpu::vulkan
