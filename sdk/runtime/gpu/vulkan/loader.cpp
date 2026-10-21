#include "gpu/vulkan/loader.h"

namespace gpu::vulkan {

Loader::~Loader() {
    Reset();
}

bool Loader::Load(std::string& error) {
    if (Loaded()) return true;
    module_ = LoadLibraryExW(L"vulkan-1.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!module_) {
        error = "Vulkan loader vulkan-1.dll is unavailable";
        return false;
    }
    get_instance_proc_addr_ = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        GetProcAddress(module_, "vkGetInstanceProcAddr"));
    if (!get_instance_proc_addr_) {
        error = "Vulkan loader is missing vkGetInstanceProcAddr";
        Reset();
        return false;
    }
    return true;
}

void Loader::Reset() {
    get_instance_proc_addr_ = nullptr;
    if (module_) FreeLibrary(module_);
    module_ = nullptr;
}

}  // namespace gpu::vulkan
