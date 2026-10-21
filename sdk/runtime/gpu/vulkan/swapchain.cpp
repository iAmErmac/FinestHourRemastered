#include "gpu/vulkan/swapchain.h"

#include "core/log.h"
#include "gpu/vulkan/context.h"
#include "gpu/vulkan/win32_surface.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <vector>

namespace gpu::vulkan {
namespace {

constexpr uint32_t kMaxSurfaceFormats = 64;
constexpr uint32_t kMaxPresentModes = 32;
constexpr uint32_t kMaxSwapchainImages = 8;
constexpr uint32_t kMaxAcquireSlots = 3;
constexpr uint64_t kDefaultRetirementTimeoutNs = 1'000'000'000ull;
constexpr uint64_t kMaxAcquireWaitNs = 500'000'000ull;

std::string ResultText(VkResult result) {
    return "VkResult " + std::to_string(static_cast<int>(result));
}

bool HasMode(const std::vector<VkPresentModeKHR>& modes, VkPresentModeKHR mode) {
    return std::find(modes.begin(), modes.end(), mode) != modes.end();
}

VkCompositeAlphaFlagBitsKHR ChooseCompositeAlpha(VkCompositeAlphaFlagsKHR supported) {
    constexpr std::array<VkCompositeAlphaFlagBitsKHR, 4> choices = {
        VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
        VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
        VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
    };
    for (VkCompositeAlphaFlagBitsKHR choice : choices)
        if (supported & choice) return choice;
    return VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
}

}  // namespace

Swapchain::~Swapchain() {
    Reset();
}

SwapchainInitStatus Swapchain::Init(const Context& context, const Win32Surface& surface, VkExtent2D extent,
                                    PresentPreference preference, std::string& error) {
    error.clear();
    if (context_ || swapchain_) {
        error = "Vulkan swapchain Init requires Shutdown before reusing a live owner";
        return SwapchainInitStatus::Failed;
    }
    if (!context.Initialized() || context.Instance() == VK_NULL_HANDLE || context.PhysicalDevice() == VK_NULL_HANDLE ||
        context.Device() == VK_NULL_HANDLE || context.GraphicsQueue() == VK_NULL_HANDLE) {
        error = "Vulkan swapchain requires an initialized Context";
        return SwapchainInitStatus::Failed;
    }
    if (surface.Handle() == VK_NULL_HANDLE || surface.Instance() != context.Instance()) {
        error = "Vulkan swapchain surface is not owned by the supplied Context instance";
        return SwapchainInitStatus::Failed;
    }
    if (!surface.SupportsGraphicsQueue(context, error)) return SwapchainInitStatus::Failed;
    if (!context.SwapchainMaintenance1Enabled()) {
        error = "Vulkan swapchain requires enabled VK_EXT_swapchain_maintenance1 present-fence retirement";
        return SwapchainInitStatus::CapabilityLimited;
    }
    context_ = &context;
    surface_ = &surface;
    device_ = context.Device();
    physical_device_ = context.PhysicalDevice();
    queue_ = context.GraphicsQueue();
    queue_family_ = context.GraphicsQueueFamily();
    surface_handle_ = surface.Handle();
    present_fence_retirement_ = true;
    if (!LoadFunctions(error)) {
        Reset();
        return SwapchainInitStatus::Failed;
    }
    const SwapchainInitStatus status = Create(extent, preference, error);
    if (status == SwapchainInitStatus::Failed || status == SwapchainInitStatus::DeviceLost ||
        status == SwapchainInitStatus::SurfaceLost) {
        Reset();
    }
    return status;
}

SwapchainInitStatus Swapchain::Recreate(VkExtent2D extent, PresentPreference preference, std::string& error) {
    error.clear();
    if (!context_ || !surface_ || !present_fence_retirement_) {
        error = "Vulkan swapchain is not initialized";
        return SwapchainInitStatus::Failed;
    }
    if (acquired_image_ != UINT32_MAX) {
        error = "Vulkan swapchain recreation requires Present or Abandon for the acquired image";
        return SwapchainInitStatus::Failed;
    }
    if (!WaitForPresentCompletion(kDefaultRetirementTimeoutNs, error)) return SwapchainInitStatus::Failed;
    DestroyResources();
    return Create(extent, preference, error);
}

bool Swapchain::Shutdown(std::string& error) {
    error.clear();
    if (!swapchain_ && !context_) return true;
    if (acquired_image_ != UINT32_MAX) {
        error = "Vulkan swapchain shutdown requires Present or Abandon for the acquired image";
        return false;
    }
    if (!WaitForPresentCompletion(kDefaultRetirementTimeoutNs, error)) return false;
    DestroyResources();
    ClearOwner();
    return true;
}

void Swapchain::Reset() {
    std::string ignored;
    if (Shutdown(ignored)) return;

    // Public Shutdown is bounded and retryable. The final owner path waits for actual present fences before
    // releasing handles; if an unrecoverable non-device-loss failure remains, terminate rather than return
    // from destruction with live Vulkan handles.
    if (acquired_image_ != UINT32_MAX) {
        AcquireResult active{};
        active.status = AcquireStatus::Acquired;
        active.owner = this;
        active.generation = generation_;
        active.image_index = acquired_image_;
        active.acquire_slot = acquired_slot_;
        active.acquire_semaphore = acquired_slot_ < acquire_slots_.size() ? acquire_slots_[acquired_slot_].semaphore
                                                                            : VK_NULL_HANDLE;
        active.acquisition_serial = acquired_serial_;
        std::string abandon_error;
        if (!AbandonInternal(active, std::numeric_limits<uint64_t>::max(), abandon_error)) {
            LOGE("vulkan: swapchain final cleanup could not abandon acquisition: %s", abandon_error.c_str());
            if (device_lost_) {
                // Device loss is terminal: release this owner's device children and never reuse the handles.
                DestroyResources();
                ClearOwner();
                return;
            }
            LOGE("vulkan: swapchain final cleanup is unrecoverable; terminating with owner state intact");
            std::abort();
        }
    }
    std::string final_error;
    if (WaitForPresentCompletion(std::numeric_limits<uint64_t>::max(), final_error)) {
        DestroyResources();
        ClearOwner();
        return;
    }
    if (device_lost_) {
        // No retryable Vulkan work remains after device loss. Do not leave dangling state after destruction.
        DestroyResources();
        ClearOwner();
        return;
    }
    LOGE("vulkan: swapchain final cleanup is unrecoverable; terminating: %s", final_error.c_str());
    std::abort();
}

bool Swapchain::LoadFunctions(std::string& error) {
#define LOAD_INSTANCE(type, field, name)                                                                    \
    field = reinterpret_cast<type>(context_->InstanceProc(name));                                           \
    if (!field) {                                                                                            \
        error = std::string("Vulkan instance is missing ") + name;                                          \
        return false;                                                                                        \
    }
#define LOAD_DEVICE(type, field, name)                                                                      \
    field = reinterpret_cast<type>(context_->DeviceProc(name));                                             \
    if (!field) {                                                                                            \
        error = std::string("Vulkan device is missing ") + name;                                            \
        return false;                                                                                        \
    }
    LOAD_INSTANCE(PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR, get_surface_capabilities_, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR")
    LOAD_INSTANCE(PFN_vkGetPhysicalDeviceSurfaceFormatsKHR, get_surface_formats_, "vkGetPhysicalDeviceSurfaceFormatsKHR")
    LOAD_INSTANCE(PFN_vkGetPhysicalDeviceSurfacePresentModesKHR, get_surface_modes_, "vkGetPhysicalDeviceSurfacePresentModesKHR")
    LOAD_DEVICE(PFN_vkCreateSwapchainKHR, create_swapchain_, "vkCreateSwapchainKHR")
    LOAD_DEVICE(PFN_vkDestroySwapchainKHR, destroy_swapchain_, "vkDestroySwapchainKHR")
    LOAD_DEVICE(PFN_vkGetSwapchainImagesKHR, get_swapchain_images_, "vkGetSwapchainImagesKHR")
    LOAD_DEVICE(PFN_vkAcquireNextImageKHR, acquire_next_image_, "vkAcquireNextImageKHR")
    LOAD_DEVICE(PFN_vkQueuePresentKHR, queue_present_, "vkQueuePresentKHR")
    LOAD_DEVICE(PFN_vkCreateImageView, create_image_view_, "vkCreateImageView")
    LOAD_DEVICE(PFN_vkDestroyImageView, destroy_image_view_, "vkDestroyImageView")
    LOAD_DEVICE(PFN_vkCreateSemaphore, create_semaphore_, "vkCreateSemaphore")
    LOAD_DEVICE(PFN_vkDestroySemaphore, destroy_semaphore_, "vkDestroySemaphore")
    LOAD_DEVICE(PFN_vkCreateFence, create_fence_, "vkCreateFence")
    LOAD_DEVICE(PFN_vkDestroyFence, destroy_fence_, "vkDestroyFence")
    LOAD_DEVICE(PFN_vkGetFenceStatus, get_fence_status_, "vkGetFenceStatus")
    LOAD_DEVICE(PFN_vkWaitForFences, wait_for_fences_, "vkWaitForFences")
    LOAD_DEVICE(PFN_vkResetFences, reset_fences_, "vkResetFences")
    LOAD_DEVICE(PFN_vkQueueSubmit2, queue_submit2_, "vkQueueSubmit2")
    release_swapchain_images_ = reinterpret_cast<PFN_vkReleaseSwapchainImagesEXT>(
        context_->DeviceProc("vkReleaseSwapchainImagesEXT"));
#undef LOAD_INSTANCE
#undef LOAD_DEVICE
    if (!release_swapchain_images_) {
        error = "Vulkan device is missing vkReleaseSwapchainImagesEXT required for aborted acquisitions";
        return false;
    }
    return true;
}

SwapchainInitStatus Swapchain::Create(VkExtent2D requested_extent, PresentPreference preference, std::string& error) {
    if (requested_extent.width == 0 || requested_extent.height == 0) {
        extent_ = {};
        return SwapchainInitStatus::Suspended;
    }

    VkSurfaceCapabilitiesKHR capabilities{};
    VkResult result = get_surface_capabilities_(physical_device_, surface_handle_, &capabilities);
    if (result == VK_ERROR_SURFACE_LOST_KHR) {
        error = "vkGetPhysicalDeviceSurfaceCapabilitiesKHR reported surface lost";
        return SwapchainInitStatus::SurfaceLost;
    }
    if (result != VK_SUCCESS) {
        error = "vkGetPhysicalDeviceSurfaceCapabilitiesKHR failed: " + ResultText(result);
        return result == VK_ERROR_DEVICE_LOST ? SwapchainInitStatus::DeviceLost : SwapchainInitStatus::Failed;
    }

    VkExtent2D actual_extent = requested_extent;
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        actual_extent = capabilities.currentExtent;
    } else {
        actual_extent.width = std::clamp(requested_extent.width, capabilities.minImageExtent.width,
                                         capabilities.maxImageExtent.width);
        actual_extent.height = std::clamp(requested_extent.height, capabilities.minImageExtent.height,
                                          capabilities.maxImageExtent.height);
    }
    if (actual_extent.width == 0 || actual_extent.height == 0) {
        extent_ = {};
        return SwapchainInitStatus::Suspended;
    }

    uint32_t format_count = 0;
    result = get_surface_formats_(physical_device_, surface_handle_, &format_count, nullptr);
    if (result != VK_SUCCESS || format_count == 0 || format_count > kMaxSurfaceFormats) {
        error = result == VK_SUCCESS ? "Vulkan surface returned no/boundedly unsupported formats"
                                     : "vkGetPhysicalDeviceSurfaceFormatsKHR failed: " + ResultText(result);
        return result == VK_ERROR_DEVICE_LOST ? SwapchainInitStatus::DeviceLost : SwapchainInitStatus::Failed;
    }
    std::vector<VkSurfaceFormatKHR> formats(format_count);
    result = get_surface_formats_(physical_device_, surface_handle_, &format_count, formats.data());
    if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
        error = "vkGetPhysicalDeviceSurfaceFormatsKHR list failed: " + ResultText(result);
        return result == VK_ERROR_DEVICE_LOST ? SwapchainInitStatus::DeviceLost : SwapchainInitStatus::Failed;
    }
    formats.resize(format_count);
    VkSurfaceFormatKHR chosen_format{VK_FORMAT_UNDEFINED, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    if (formats.empty()) {
        error = "Vulkan surface returned no formats after bounded enumeration";
        return SwapchainInitStatus::Failed;
    }
    if (formats.size() == 1 && formats[0].format == VK_FORMAT_UNDEFINED) {
        chosen_format = {VK_FORMAT_B8G8R8A8_UNORM, formats[0].colorSpace};
    } else {
        const std::array<VkSurfaceFormatKHR, 2> preferred = {
            VkSurfaceFormatKHR{VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR},
            VkSurfaceFormatKHR{VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR},
        };
        bool found = false;
        for (const VkSurfaceFormatKHR& candidate : preferred) {
            auto it = std::find_if(formats.begin(), formats.end(), [&candidate](const VkSurfaceFormatKHR& value) {
                return value.format == candidate.format && value.colorSpace == candidate.colorSpace;
            });
            if (it != formats.end()) {
                chosen_format = *it;
                found = true;
                break;
            }
        }
        if (!found) chosen_format = formats.front();
    }

    uint32_t mode_count = 0;
    result = get_surface_modes_(physical_device_, surface_handle_, &mode_count, nullptr);
    if (result != VK_SUCCESS || mode_count == 0 || mode_count > kMaxPresentModes) {
        error = result == VK_SUCCESS ? "Vulkan surface returned no/boundedly unsupported present modes"
                                     : "vkGetPhysicalDeviceSurfacePresentModesKHR failed: " + ResultText(result);
        return result == VK_ERROR_DEVICE_LOST ? SwapchainInitStatus::DeviceLost : SwapchainInitStatus::Failed;
    }
    std::vector<VkPresentModeKHR> modes(mode_count);
    result = get_surface_modes_(physical_device_, surface_handle_, &mode_count, modes.data());
    if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
        error = "vkGetPhysicalDeviceSurfacePresentModesKHR list failed: " + ResultText(result);
        return result == VK_ERROR_DEVICE_LOST ? SwapchainInitStatus::DeviceLost : SwapchainInitStatus::Failed;
    }
    modes.resize(mode_count);
    if (!HasMode(modes, VK_PRESENT_MODE_FIFO_KHR)) {
        error = "Vulkan surface lacks the required FIFO present mode";
        return SwapchainInitStatus::Failed;
    }

    VkImageUsageFlags required_usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if ((capabilities.supportedUsageFlags & required_usage) != required_usage) {
        error = "Vulkan surface lacks color-attachment swapchain usage";
        return SwapchainInitStatus::Failed;
    }
    // Transfer-source usage is optional for the runtime, but enables swapchain-image readback/capture when
    // the surface advertises it. Do not reject an otherwise usable surface.
    if (capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
        required_usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    const uint32_t desired_images = capabilities.minImageCount + 1;
    const uint32_t image_count = capabilities.maxImageCount
                                     ? std::min(desired_images, capabilities.maxImageCount)
                                     : desired_images;
    if (image_count == 0 || image_count > kMaxSwapchainImages) {
        error = "Vulkan surface image count exceeds the bounded WSI resource limit";
        return SwapchainInitStatus::Failed;
    }

    VkSurfaceTransformFlagBitsKHR transform =
        (capabilities.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
            ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
            : capabilities.currentTransform;
    const VkCompositeAlphaFlagBitsKHR composite_alpha = ChooseCompositeAlpha(capabilities.supportedCompositeAlpha);
    const VkPresentModeKHR present_mode = ChoosePresentMode(modes, preference);
    VkSwapchainCreateInfoKHR create{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    create.surface = surface_handle_;
    create.minImageCount = image_count;
    create.imageFormat = chosen_format.format;
    create.imageColorSpace = chosen_format.colorSpace;
    create.imageExtent = actual_extent;
    create.imageArrayLayers = 1;
    create.imageUsage = required_usage;
    create.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    create.preTransform = transform;
    create.compositeAlpha = composite_alpha;
    create.presentMode = present_mode;
    create.clipped = VK_TRUE;

    VkSwapchainKHR fresh = VK_NULL_HANDLE;
    result = create_swapchain_(device_, &create, nullptr, &fresh);
    if (result == VK_ERROR_SURFACE_LOST_KHR) {
        error = "vkCreateSwapchainKHR reported surface lost";
        return SwapchainInitStatus::SurfaceLost;
    }
    if (result == VK_ERROR_DEVICE_LOST) {
        error = "vkCreateSwapchainKHR reported device lost";
        return SwapchainInitStatus::DeviceLost;
    }
    if (result != VK_SUCCESS) {
        error = "vkCreateSwapchainKHR failed: " + ResultText(result);
        return SwapchainInitStatus::Failed;
    }

    uint32_t actual_count = 0;
    result = get_swapchain_images_(device_, fresh, &actual_count, nullptr);
    if (result != VK_SUCCESS || actual_count == 0 || actual_count > kMaxSwapchainImages) {
        destroy_swapchain_(device_, fresh, nullptr);
        error = result == VK_SUCCESS ? "Vulkan swapchain returned an unsupported image count"
                                     : "vkGetSwapchainImagesKHR count failed: " + ResultText(result);
        return result == VK_ERROR_DEVICE_LOST ? SwapchainInitStatus::DeviceLost : SwapchainInitStatus::Failed;
    }
    std::vector<VkImage> fresh_images(actual_count);
    result = get_swapchain_images_(device_, fresh, &actual_count, fresh_images.data());
    if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
        destroy_swapchain_(device_, fresh, nullptr);
        error = "vkGetSwapchainImagesKHR list failed: " + ResultText(result);
        return result == VK_ERROR_DEVICE_LOST ? SwapchainInitStatus::DeviceLost : SwapchainInitStatus::Failed;
    }
    fresh_images.resize(actual_count);
    if (fresh_images.empty() || fresh_images.size() > kMaxSwapchainImages) {
        destroy_swapchain_(device_, fresh, nullptr);
        error = "Vulkan swapchain image enumeration exceeded the bounded limit";
        return SwapchainInitStatus::Failed;
    }

    std::vector<VkImageView> fresh_views;
    std::vector<VkSemaphore> fresh_present_semaphores;
    std::vector<VkFence> fresh_present_fences;
    std::vector<VkFence> fresh_render_fences;
    fresh_views.reserve(fresh_images.size());
    fresh_present_semaphores.reserve(fresh_images.size());
    fresh_present_fences.reserve(fresh_images.size());
    fresh_render_fences.reserve(fresh_images.size());
    auto cleanup_fresh = [&]() {
        for (VkFence fence : fresh_render_fences) destroy_fence_(device_, fence, nullptr);
        for (VkFence fence : fresh_present_fences) destroy_fence_(device_, fence, nullptr);
        for (VkSemaphore semaphore : fresh_present_semaphores) destroy_semaphore_(device_, semaphore, nullptr);
        for (VkImageView view : fresh_views) destroy_image_view_(device_, view, nullptr);
        destroy_swapchain_(device_, fresh, nullptr);
    };
    for (VkImage image : fresh_images) {
        VkImageViewCreateInfo view_create{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_create.image = image;
        view_create.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_create.format = chosen_format.format;
        view_create.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkImageView view = VK_NULL_HANDLE;
        result = create_image_view_(device_, &view_create, nullptr, &view);
        if (result != VK_SUCCESS) {
            cleanup_fresh();
            error = "vkCreateImageView failed: " + ResultText(result);
            return result == VK_ERROR_DEVICE_LOST ? SwapchainInitStatus::DeviceLost : SwapchainInitStatus::Failed;
        }
        fresh_views.push_back(view);

        VkSemaphoreCreateInfo semaphore_create{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkSemaphore semaphore = VK_NULL_HANDLE;
        result = create_semaphore_(device_, &semaphore_create, nullptr, &semaphore);
        if (result != VK_SUCCESS) {
            cleanup_fresh();
            error = "vkCreateSemaphore failed: " + ResultText(result);
            return result == VK_ERROR_DEVICE_LOST ? SwapchainInitStatus::DeviceLost : SwapchainInitStatus::Failed;
        }
        fresh_present_semaphores.push_back(semaphore);

        VkFenceCreateInfo fence_create{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fence_create.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VkFence fence = VK_NULL_HANDLE;
        result = create_fence_(device_, &fence_create, nullptr, &fence);
        if (result != VK_SUCCESS) {
            cleanup_fresh();
            error = "vkCreateFence failed: " + ResultText(result);
            return result == VK_ERROR_DEVICE_LOST ? SwapchainInitStatus::DeviceLost : SwapchainInitStatus::Failed;
        }
        fresh_present_fences.push_back(fence);

        VkFence render_fence = VK_NULL_HANDLE;
        result = create_fence_(device_, &fence_create, nullptr, &render_fence);
        if (result != VK_SUCCESS) {
            cleanup_fresh();
            error = "vkCreateFence for rendering failed: " + ResultText(result);
            return result == VK_ERROR_DEVICE_LOST ? SwapchainInitStatus::DeviceLost : SwapchainInitStatus::Failed;
        }
        fresh_render_fences.push_back(render_fence);
    }

    const uint32_t acquire_slot_count = std::min<uint32_t>(kMaxAcquireSlots, actual_count);
    std::vector<AcquireSlot> fresh_acquire_slots(acquire_slot_count);
    for (AcquireSlot& slot : fresh_acquire_slots) {
        VkSemaphoreCreateInfo semaphore_create{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        result = create_semaphore_(device_, &semaphore_create, nullptr, &slot.semaphore);
        if (result != VK_SUCCESS) {
            for (AcquireSlot& created : fresh_acquire_slots)
                if (created.semaphore) destroy_semaphore_(device_, created.semaphore, nullptr);
            cleanup_fresh();
            error = "vkCreateSemaphore for acquire failed: " + ResultText(result);
            return result == VK_ERROR_DEVICE_LOST ? SwapchainInitStatus::DeviceLost : SwapchainInitStatus::Failed;
        }
    }

    swapchain_ = fresh;
    format_ = chosen_format.format;
    color_space_ = chosen_format.colorSpace;
    extent_ = actual_extent;
    image_usage_ = required_usage;
    pre_transform_ = transform;
    composite_alpha_ = composite_alpha;
    present_mode_ = present_mode;
    images_ = std::move(fresh_images);
    views_ = std::move(fresh_views);
    present_semaphores_ = std::move(fresh_present_semaphores);
    present_fences_ = std::move(fresh_present_fences);
    render_fences_ = std::move(fresh_render_fences);
    present_pending_.assign(images_.size(), false);
    present_fence_armed_.assign(images_.size(), false);
    render_pending_.assign(images_.size(), false);
    acquire_slots_ = std::move(fresh_acquire_slots);
    acquired_image_ = UINT32_MAX;
    acquired_slot_ = UINT32_MAX;
    acquired_serial_ = 0;
    submission_submitted_ = false;
    cleanup_wait_consumed_ = false;
    ++generation_;
    LOGI("vulkan: swapchain %ux%u, %u images, present mode %d, present fences on", extent_.width, extent_.height,
         static_cast<unsigned>(images_.size()), static_cast<int>(present_mode_));
    return SwapchainInitStatus::Ready;
}

bool Swapchain::WaitForPresentCompletion(uint64_t timeout_ns, std::string& error) {
    if (!swapchain_) return true;
    if (!WaitForRendering(timeout_ns, error)) return false;
    for (uint32_t i = 0; i < present_pending_.size(); ++i) {
        if (!present_pending_[i]) continue;
        const VkResult result = wait_for_fences_(device_, 1, &present_fences_[i], VK_TRUE, timeout_ns);
        if (result != VK_SUCCESS) {
            MarkDeviceLost(result);
            error = "present fence did not complete: " + ResultText(result);
            return false;
        }
        present_pending_[i] = false;
        present_fence_armed_[i] = false;
    }
    for (AcquireSlot& slot : acquire_slots_) slot.in_flight = false;
    return true;
}

bool Swapchain::WaitForImagePresent(uint32_t image_index, uint64_t timeout_ns, std::string& error) {
    if (image_index >= present_pending_.size()) return false;
    if (!WaitForRenderCompletion(image_index, timeout_ns, error)) return false;
    if (!present_pending_[image_index]) return true;
    const VkResult result = wait_for_fences_(device_, 1, &present_fences_[image_index], VK_TRUE, timeout_ns);
    if (result != VK_SUCCESS) {
        MarkDeviceLost(result);
        error = "acquired image's present fence did not complete: " + ResultText(result);
        return false;
    }
    present_pending_[image_index] = false;
    present_fence_armed_[image_index] = false;
    for (AcquireSlot& slot : acquire_slots_)
        if (slot.image_index == image_index) slot.in_flight = false;
    return true;
}

bool Swapchain::WaitForRenderCompletion(uint32_t image_index, uint64_t timeout_ns, std::string& error) {
    if (image_index >= render_pending_.size() || !render_pending_[image_index]) return true;
    const VkResult result = wait_for_fences_(device_, 1, &render_fences_[image_index], VK_TRUE, timeout_ns);
    if (result != VK_SUCCESS) {
        MarkDeviceLost(result);
        error = "render fence did not complete: " + ResultText(result);
        return false;
    }
    render_pending_[image_index] = false;
    return true;
}

bool Swapchain::WaitForRendering(uint64_t timeout_ns, std::string& error) {
    error.clear();
    if (!swapchain_) return true;
    for (uint32_t i = 0; i < render_pending_.size(); ++i)
        if (!WaitForRenderCompletion(i, timeout_ns, error)) return false;
    return true;
}

bool Swapchain::FindAcquireSlot(uint64_t timeout_ns, uint32_t& slot, std::string& error) {
    for (uint32_t i = 0; i < acquire_slots_.size(); ++i) {
        AcquireSlot& candidate = acquire_slots_[i];
        if (!candidate.in_flight) {
            slot = i;
            return true;
        }
        if (candidate.image_index < present_pending_.size()) {
            const VkResult status = get_fence_status_(device_, present_fences_[candidate.image_index]);
            if (status == VK_SUCCESS) {
                if (!WaitForRenderCompletion(candidate.image_index, timeout_ns, error)) return false;
                present_pending_[candidate.image_index] = false;
                present_fence_armed_[candidate.image_index] = false;
                candidate.in_flight = false;
                slot = i;
                return true;
            }
            if (status != VK_NOT_READY) {
                MarkDeviceLost(status);
                error = "vkGetFenceStatus failed: " + ResultText(status);
                return false;
            }
        }
    }
    if (timeout_ns == 0 || acquire_slots_.empty()) return false;
    uint32_t wait_slot = UINT32_MAX;
    for (uint32_t i = 0; i < acquire_slots_.size(); ++i)
        if (acquire_slots_[i].in_flight) {
            wait_slot = i;
            break;
        }
    if (wait_slot == UINT32_MAX) return false;
    const uint64_t wait_ns = std::min(timeout_ns, kMaxAcquireWaitNs);
    const uint32_t image = acquire_slots_[wait_slot].image_index;
    if (!WaitForImagePresent(image, wait_ns, error)) return false;
    acquire_slots_[wait_slot].in_flight = false;
    slot = wait_slot;
    return true;
}

AcquireResult Swapchain::Acquire(uint64_t timeout_ns, std::string& error) {
    error.clear();
    AcquireResult result{};
    result.owner = this;
    result.generation = generation_;
    if (!swapchain_) {
        result.status = AcquireStatus::Suspended;
        return result;
    }
    if (acquired_image_ != UINT32_MAX) {
        error = "Vulkan swapchain Acquire called while an image is already owned";
        result.status = AcquireStatus::Failed;
        return result;
    }
    const uint64_t budget = std::min(timeout_ns, kMaxAcquireWaitNs);
    const uint64_t slot_budget = budget / 2;
    const uint64_t image_budget = budget - slot_budget;
    uint32_t slot = UINT32_MAX;
    if (!FindAcquireSlot(slot_budget, slot, error)) {
        result.status = error.empty() ? (timeout_ns == 0 ? AcquireStatus::NotReady : AcquireStatus::Timeout)
                                      : AcquireStatus::Failed;
        return result;
    }
    uint32_t image_index = UINT32_MAX;
    const VkResult acquire_result = acquire_next_image_(device_, swapchain_, image_budget, acquire_slots_[slot].semaphore,
                                                        VK_NULL_HANDLE, &image_index);
    if (acquire_result == VK_TIMEOUT) {
        result.status = AcquireStatus::Timeout;
        return result;
    }
    if (acquire_result == VK_NOT_READY) {
        result.status = AcquireStatus::NotReady;
        return result;
    }
    if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR) {
        result.status = AcquireStatus::RecreateRequired;
        return result;
    }
    if (acquire_result == VK_ERROR_SURFACE_LOST_KHR) {
        result.status = AcquireStatus::SurfaceLost;
        error = "vkAcquireNextImageKHR reported surface lost";
        return result;
    }
    if (acquire_result == VK_ERROR_DEVICE_LOST) {
        MarkDeviceLost(acquire_result);
        result.status = AcquireStatus::DeviceLost;
        error = "vkAcquireNextImageKHR reported device lost";
        return result;
    }
    if (acquire_result != VK_SUCCESS && acquire_result != VK_SUBOPTIMAL_KHR) {
        MarkDeviceLost(acquire_result);
        result.status = AcquireStatus::Failed;
        error = "vkAcquireNextImageKHR failed: " + ResultText(acquire_result);
        return result;
    }
    if (image_index >= images_.size()) {
        error = "vkAcquireNextImageKHR returned an out-of-range image index";
        result.status = AcquireStatus::Failed;
        return result;
    }
    acquire_slots_[slot].image_index = image_index;
    acquired_image_ = image_index;
    acquired_slot_ = slot;
    acquired_serial_ = ++acquisition_serial_;
    result.status = AcquireStatus::Acquired;
    result.image_index = image_index;
    result.acquire_slot = slot;
    result.acquire_semaphore = acquire_slots_[slot].semaphore;
    result.acquisition_serial = acquired_serial_;
    result.suboptimal = acquire_result == VK_SUBOPTIMAL_KHR;
    cleanup_wait_consumed_ = false;
    submission_submitted_ = false;
    if (!WaitForImagePresent(image_index, image_budget, error)) {
        const std::string wait_error = error;
        std::string abandon_error;
        const bool abandoned = Abandon(result, abandon_error);
        if (!abandoned) {
            error = wait_error + "; acquire abort failed: " + abandon_error;
        } else {
            error = wait_error;
        }
        result.status = abandoned ? AcquireStatus::Timeout : AcquireStatus::RecoveryRequired;
        return result;
    }
    return result;
}

bool Swapchain::ValidateResult(const AcquireResult& result, std::string& error) const {
    if (result.status != AcquireStatus::Acquired || result.owner != this || result.generation != generation_ ||
        result.acquisition_serial == 0 || result.acquisition_serial != acquired_serial_ ||
        result.image_index >= images_.size() ||
        result.acquire_slot >= acquire_slots_.size() || result.image_index != acquired_image_ ||
        result.acquire_slot != acquired_slot_ || result.acquire_semaphore != acquire_slots_[result.acquire_slot].semaphore) {
        error = "Vulkan swapchain acquire token is stale, foreign or already consumed";
        return false;
    }
    return true;
}

bool Swapchain::ValidateAbandonResult(const AcquireResult& result, std::string& error) const {
    if ((result.status != AcquireStatus::Acquired && result.status != AcquireStatus::RecoveryRequired) ||
        result.owner != this || result.generation != generation_ || result.acquisition_serial == 0 ||
        result.acquisition_serial != acquired_serial_ || result.image_index >= images_.size() ||
        result.acquire_slot >= acquire_slots_.size() || result.image_index != acquired_image_ ||
        result.acquire_slot != acquired_slot_ ||
        result.acquire_semaphore != acquire_slots_[result.acquire_slot].semaphore) {
        error = "Vulkan swapchain acquire recovery token is stale, foreign or already consumed";
        return false;
    }
    return true;
}

bool Swapchain::Submit(const AcquireResult& result, VkCommandBuffer command_buffer, std::string& error) {
    error.clear();
    if (!ValidateResult(result, error)) return false;
    if (submission_submitted_) {
        error = "Vulkan swapchain Submit was already called for this acquired image";
        return false;
    }
    if (command_buffer == VK_NULL_HANDLE) {
        error = "Vulkan swapchain Submit requires a command buffer";
        return false;
    }
    VkFence render_fence = render_fences_[result.image_index];
    if (render_fence == VK_NULL_HANDLE) {
        error = "Vulkan swapchain is missing its reusable render fence";
        return false;
    }
    const VkResult reset_result = reset_fences_(device_, 1, &render_fence);
    if (reset_result != VK_SUCCESS) {
        MarkDeviceLost(reset_result);
        error = "vkResetFences for rendering failed: " + ResultText(reset_result);
        return false;
    }
    VkSemaphoreSubmitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    wait.semaphore = result.acquire_semaphore;
    wait.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkCommandBufferSubmitInfo command{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    command.commandBuffer = command_buffer;
    VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    signal.semaphore = present_semaphores_[result.image_index];
    signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.waitSemaphoreInfoCount = 1;
    submit.pWaitSemaphoreInfos = &wait;
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &command;
    submit.signalSemaphoreInfoCount = 1;
    submit.pSignalSemaphoreInfos = &signal;
    const VkResult submit_result = queue_submit2_(queue_, 1, &submit, render_fence);
    if (submit_result != VK_SUCCESS) {
        MarkDeviceLost(submit_result);
        error = "vkQueueSubmit2 for rendering failed: " + ResultText(submit_result);
        return false;
    }
    render_pending_[result.image_index] = true;
    submission_submitted_ = true;
    return true;
}

VkFence Swapchain::RenderFence(const AcquireResult& result) const {
    if (result.owner != this || result.image_index >= render_fences_.size() || !submission_submitted_)
        return VK_NULL_HANDLE;
    return render_fences_[result.image_index];
}

PresentStatus Swapchain::Present(const AcquireResult& result, std::string& error) {
    error.clear();
    if (!ValidateResult(result, error)) return PresentStatus::Failed;
    if (!submission_submitted_) {
        error = "Vulkan swapchain Present requires Submit for the acquired image";
        return PresentStatus::Failed;
    }
    if (!present_fence_armed_[result.image_index]) {
        const VkResult reset_result = reset_fences_(device_, 1, &present_fences_[result.image_index]);
        if (reset_result != VK_SUCCESS) {
            MarkDeviceLost(reset_result);
            error = "vkResetFences for presentation failed: " + ResultText(reset_result);
            return reset_result == VK_ERROR_DEVICE_LOST ? PresentStatus::DeviceLost : PresentStatus::Failed;
        }
        present_fence_armed_[result.image_index] = true;
    }
    VkSwapchainPresentFenceInfoEXT fence_info{VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_EXT};
    fence_info.swapchainCount = 1;
    fence_info.pFences = &present_fences_[result.image_index];
    VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.pNext = &fence_info;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &present_semaphores_[result.image_index];
    present.swapchainCount = 1;
    present.pSwapchains = &swapchain_;
    present.pImageIndices = &result.image_index;
    const VkResult present_result = queue_present_(queue_, &present);
    const bool enqueued = present_result == VK_SUCCESS || present_result == VK_SUBOPTIMAL_KHR ||
                          present_result == VK_ERROR_OUT_OF_DATE_KHR || present_result == VK_ERROR_SURFACE_LOST_KHR ||
                          present_result == VK_ERROR_DEVICE_LOST;
    if (!enqueued) {
        // Host/device allocation errors leave the wait semaphore and acquire token available for a bounded retry.
        MarkDeviceLost(present_result);
        error = "vkQueuePresentKHR did not enqueue presentation: " + ResultText(present_result);
        return present_result == VK_ERROR_DEVICE_LOST ? PresentStatus::DeviceLost : PresentStatus::Failed;
    }
    // Acquire waited for the previous fence. The armed per-image fence now covers this present attempt.
    acquired_image_ = UINT32_MAX;
    acquired_slot_ = UINT32_MAX;
    acquired_serial_ = 0;
    submission_submitted_ = false;
    acquire_slots_[result.acquire_slot].in_flight = true;
    present_pending_[result.image_index] = true;
    ++presented_count_;
    if (present_result == VK_SUCCESS) return PresentStatus::Presented;
    if (present_result == VK_SUBOPTIMAL_KHR || present_result == VK_ERROR_OUT_OF_DATE_KHR) return PresentStatus::RecreateRequired;
    if (present_result == VK_ERROR_SURFACE_LOST_KHR) {
        error = "vkQueuePresentKHR reported surface lost";
        return PresentStatus::SurfaceLost;
    }
    if (present_result == VK_ERROR_DEVICE_LOST) {
        device_lost_ = true;
        error = "vkQueuePresentKHR reported device lost";
        return PresentStatus::DeviceLost;
    }
    error = "vkQueuePresentKHR failed: " + ResultText(present_result);
    return PresentStatus::Failed;
}

bool Swapchain::SubmitSemaphoreWait(VkSemaphore semaphore, uint64_t timeout_ns, uint64_t serial, std::string& error) {
    if (pending_cleanup_fence_ != VK_NULL_HANDLE) {
        if (pending_cleanup_serial_ != serial) {
            error = "another semaphore cleanup is still pending completion";
            return false;
        }
        const VkResult pending_result = wait_for_fences_(device_, 1, &pending_cleanup_fence_, VK_TRUE, timeout_ns);
        if (pending_result != VK_SUCCESS) {
            MarkDeviceLost(pending_result);
            error = "semaphore cleanup wait is still pending: " + ResultText(pending_result);
            return false;
        }
        destroy_fence_(device_, pending_cleanup_fence_, nullptr);
        pending_cleanup_fence_ = VK_NULL_HANDLE;
        pending_cleanup_serial_ = 0;
        cleanup_wait_consumed_ = true;
        return true;
    }
    if (cleanup_wait_consumed_) return true;
    VkSemaphoreSubmitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    wait.semaphore = semaphore;
    wait.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.waitSemaphoreInfoCount = 1;
    submit.pWaitSemaphoreInfos = &wait;
    VkFence fence = VK_NULL_HANDLE;
    VkFenceCreateInfo fence_create{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkResult result_code = create_fence_(device_, &fence_create, nullptr, &fence);
    if (result_code != VK_SUCCESS) {
        MarkDeviceLost(result_code);
        error = "vkCreateFence for semaphore cleanup failed: " + ResultText(result_code);
        return false;
    }
    result_code = queue_submit2_(queue_, 1, &submit, fence);
    if (result_code == VK_SUCCESS) {
        result_code = wait_for_fences_(device_, 1, &fence, VK_TRUE, timeout_ns);
        if (result_code != VK_SUCCESS) {
            // A successful queue submission keeps the fence in use for every non-terminal wait error, not only timeout.
            pending_cleanup_fence_ = fence;
            pending_cleanup_serial_ = serial;
            MarkDeviceLost(result_code);
            error = "semaphore cleanup wait did not complete: " + ResultText(result_code);
            return false;
        }
    }
    destroy_fence_(device_, fence, nullptr);
    if (result_code != VK_SUCCESS) {
        MarkDeviceLost(result_code);
        error = "semaphore cleanup could not be submitted: " + ResultText(result_code);
        return false;
    }
    cleanup_wait_consumed_ = true;
    return true;
}

bool Swapchain::Abandon(const AcquireResult& result, std::string& error) {
    error.clear();
    return AbandonInternal(result, kMaxAcquireWaitNs, error);
}

bool Swapchain::AbandonInternal(const AcquireResult& result, uint64_t timeout_ns, std::string& error) {
    if (!ValidateAbandonResult(result, error)) return false;
    if (submission_submitted_) {
        if (!WaitForRenderCompletion(result.image_index, timeout_ns, error)) return false;
        if (!SubmitSemaphoreWait(present_semaphores_[result.image_index], timeout_ns, result.acquisition_serial, error))
            return false;
    } else if (!SubmitSemaphoreWait(result.acquire_semaphore, timeout_ns, result.acquisition_serial, error)) {
        return false;
    }
    VkReleaseSwapchainImagesInfoEXT release{VK_STRUCTURE_TYPE_RELEASE_SWAPCHAIN_IMAGES_INFO_EXT};
    release.swapchain = swapchain_;
    release.imageIndexCount = 1;
    release.pImageIndices = &result.image_index;
    const VkResult release_result = release_swapchain_images_(device_, &release);
    if (release_result != VK_SUCCESS) {
        MarkDeviceLost(release_result);
        error = "vkReleaseSwapchainImagesEXT failed: " + ResultText(release_result);
        return false;
    }
    acquired_image_ = UINT32_MAX;
    acquired_slot_ = UINT32_MAX;
    acquired_serial_ = 0;
    submission_submitted_ = false;
    acquire_slots_[result.acquire_slot].in_flight = false;
    acquire_slots_[result.acquire_slot].image_index = UINT32_MAX;
    cleanup_wait_consumed_ = false;
    return true;
}

VkImage Swapchain::Image(const AcquireResult& result) const {
    std::string ignored;
    if (!ValidateResult(result, ignored) || result.image_index >= images_.size())
        return VK_NULL_HANDLE;
    return images_[result.image_index];
}

VkImageView Swapchain::View(const AcquireResult& result) const {
    std::string ignored;
    if (!ValidateResult(result, ignored) || result.image_index >= views_.size())
        return VK_NULL_HANDLE;
    return views_[result.image_index];
}

VkSemaphore Swapchain::PresentSemaphore(const AcquireResult& result) const {
    std::string ignored;
    if (!ValidateResult(result, ignored) || result.image_index >= present_semaphores_.size())
        return VK_NULL_HANDLE;
    return present_semaphores_[result.image_index];
}

void Swapchain::DestroyResources() {
    if (pending_cleanup_fence_)
        destroy_fence_(device_, pending_cleanup_fence_, nullptr);
    pending_cleanup_fence_ = VK_NULL_HANDLE;
    pending_cleanup_serial_ = 0;
    for (AcquireSlot& slot : acquire_slots_)
        if (slot.semaphore) destroy_semaphore_(device_, slot.semaphore, nullptr);
    for (VkFence fence : present_fences_)
        if (fence) destroy_fence_(device_, fence, nullptr);
    for (VkFence fence : render_fences_)
        if (fence) destroy_fence_(device_, fence, nullptr);
    for (VkSemaphore semaphore : present_semaphores_)
        if (semaphore) destroy_semaphore_(device_, semaphore, nullptr);
    for (VkImageView view : views_)
        if (view) destroy_image_view_(device_, view, nullptr);
    if (swapchain_) destroy_swapchain_(device_, swapchain_, nullptr);
    acquire_slots_.clear();
    present_fences_.clear();
    render_fences_.clear();
    present_semaphores_.clear();
    present_pending_.clear();
    present_fence_armed_.clear();
    render_pending_.clear();
    views_.clear();
    images_.clear();
    swapchain_ = VK_NULL_HANDLE;
    extent_ = {};
    format_ = VK_FORMAT_UNDEFINED;
    image_usage_ = 0;
    acquired_image_ = UINT32_MAX;
    acquired_slot_ = UINT32_MAX;
    acquired_serial_ = 0;
    cleanup_wait_consumed_ = false;
    submission_submitted_ = false;
}

void Swapchain::MarkDeviceLost(VkResult result) {
    if (result == VK_ERROR_DEVICE_LOST) device_lost_ = true;
}

void Swapchain::ClearOwner() {
    context_ = nullptr;
    surface_ = nullptr;
    device_ = VK_NULL_HANDLE;
    physical_device_ = VK_NULL_HANDLE;
    queue_ = VK_NULL_HANDLE;
    queue_family_ = VK_QUEUE_FAMILY_IGNORED;
    surface_handle_ = VK_NULL_HANDLE;
    present_fence_retirement_ = false;
    device_lost_ = false;
}

VkPresentModeKHR Swapchain::ChoosePresentMode(const std::vector<VkPresentModeKHR>& modes,
                                               PresentPreference preference) {
    if (preference == PresentPreference::Mailbox && HasMode(modes, VK_PRESENT_MODE_MAILBOX_KHR))
        return VK_PRESENT_MODE_MAILBOX_KHR;
    if (preference == PresentPreference::Immediate && HasMode(modes, VK_PRESENT_MODE_IMMEDIATE_KHR))
        return VK_PRESENT_MODE_IMMEDIATE_KHR;
    return VK_PRESENT_MODE_FIFO_KHR;
}

}  // namespace gpu::vulkan
