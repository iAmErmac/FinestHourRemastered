#pragma once

#define VK_USE_PLATFORM_WIN32_KHR
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <string>

namespace gpu::vulkan {

class Context;

class Win32Surface {
public:
    Win32Surface() = default;
    Win32Surface(const Win32Surface&) = delete;
    Win32Surface& operator=(const Win32Surface&) = delete;
    ~Win32Surface();

    bool Init(const Context& context, HWND window, std::string& error);
    // The surface must be reset before its Context is destroyed.
    bool SupportsGraphicsQueue(const Context& context, std::string& error) const;
    void Reset();
    VkSurfaceKHR Handle() const { return surface_; }
    VkInstance Instance() const { return instance_; }

private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    PFN_vkDestroySurfaceKHR destroy_surface_ = nullptr;
};

}  // namespace gpu::vulkan
