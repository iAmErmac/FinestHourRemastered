#pragma once

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

class Loader {
public:
    Loader() = default;
    Loader(const Loader&) = delete;
    Loader& operator=(const Loader&) = delete;
    ~Loader();

    bool Load(std::string& error);
    void Reset();
    bool Loaded() const { return module_ != nullptr && get_instance_proc_addr_ != nullptr; }

    PFN_vkGetInstanceProcAddr GetInstanceProcAddr() const { return get_instance_proc_addr_; }

    template <typename T>
    T Export(const char* name) const {
        if (!module_) return nullptr;
        return reinterpret_cast<T>(GetProcAddress(module_, name));
    }

    template <typename T>
    T Global(const char* name) const {
        if (!get_instance_proc_addr_) return nullptr;
        return reinterpret_cast<T>(get_instance_proc_addr_(VK_NULL_HANDLE, name));
    }

private:
    HMODULE module_ = nullptr;
    PFN_vkGetInstanceProcAddr get_instance_proc_addr_ = nullptr;
};

}  // namespace gpu::vulkan
