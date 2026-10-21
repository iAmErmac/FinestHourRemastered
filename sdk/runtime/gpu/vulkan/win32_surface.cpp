#include "gpu/vulkan/win32_surface.h"

#include "gpu/vulkan/context.h"

namespace gpu::vulkan {

Win32Surface::~Win32Surface() {
    Reset();
}

bool Win32Surface::Init(const Context& context, HWND window, std::string& error) {
    Reset();
    if (context.Instance() == VK_NULL_HANDLE || !window) {
        error = "Vulkan Win32 surface requires an initialized instance and window";
        return false;
    }
    auto create_surface = reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(
        context.InstanceProc("vkCreateWin32SurfaceKHR"));
    destroy_surface_ = reinterpret_cast<PFN_vkDestroySurfaceKHR>(context.InstanceProc("vkDestroySurfaceKHR"));
    if (!create_surface || !destroy_surface_) {
        error = "Vulkan instance is missing Win32 surface entry points";
        destroy_surface_ = nullptr;
        return false;
    }
    VkWin32SurfaceCreateInfoKHR create{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
    create.hinstance = GetModuleHandleW(nullptr);
    create.hwnd = window;
    const VkResult result = create_surface(context.Instance(), &create, nullptr, &surface_);
    if (result != VK_SUCCESS) {
        error = "vkCreateWin32SurfaceKHR failed: VkResult " + std::to_string(static_cast<int>(result));
        surface_ = VK_NULL_HANDLE;
        destroy_surface_ = nullptr;
        return false;
    }
    instance_ = context.Instance();
    return true;
}

bool Win32Surface::SupportsGraphicsQueue(const Context& context, std::string& error) const {
    if (surface_ == VK_NULL_HANDLE || instance_ != context.Instance()) {
        error = "Vulkan Win32 surface is not owned by the supplied context";
        return false;
    }
    auto surface_support = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceSupportKHR>(
        context.InstanceProc("vkGetPhysicalDeviceSurfaceSupportKHR"));
    if (!surface_support) {
        error = "Vulkan instance is missing vkGetPhysicalDeviceSurfaceSupportKHR";
        return false;
    }
    VkBool32 supported = VK_FALSE;
    const VkResult result = surface_support(context.PhysicalDevice(), context.GraphicsQueueFamily(), surface_, &supported);
    if (result != VK_SUCCESS) {
        error = "vkGetPhysicalDeviceSurfaceSupportKHR failed: VkResult " + std::to_string(static_cast<int>(result));
        return false;
    }
    if (!supported) {
        error = "Selected Vulkan graphics queue does not present to the Win32 surface";
        return false;
    }
    return true;
}

void Win32Surface::Reset() {
    if (surface_ != VK_NULL_HANDLE && destroy_surface_ && instance_ != VK_NULL_HANDLE)
        destroy_surface_(instance_, surface_, nullptr);
    surface_ = VK_NULL_HANDLE;
    instance_ = VK_NULL_HANDLE;
    destroy_surface_ = nullptr;
}

}  // namespace gpu::vulkan
