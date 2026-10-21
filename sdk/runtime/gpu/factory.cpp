#include "gpu/backend.h"
#include "gpu/vulkan/backend.h"

#include "core/log.h"

#include <string>
#include <string_view>

#include <windows.h>
#include <shellapi.h>

namespace gpu {

std::unique_ptr<Backend> CreateBackend() {
    int argc = 0;
    LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!argv) {
        LOGE("renderer: Windows command-line parsing failed (%lu)", GetLastError());
        return nullptr;
    }
    enum class Choice { D3D12, Vulkan } choice = Choice::D3D12;
    bool specified = false;
    bool valid = true;
    for (int i = 1; i < argc; ++i) {
        if (std::wstring_view(argv[i]) != L"--renderer") continue;
        if (specified || i + 1 >= argc) {
            LOGE("renderer: --renderer must appear once with exactly d3d12 or vulkan");
            valid = false;
            break;
        }
        specified = true;
        const std::wstring_view value(argv[++i]);
        if (value == L"d3d12") choice = Choice::D3D12;
        else if (value == L"vulkan") choice = Choice::Vulkan;
        else {
            LOGE("renderer: unknown --renderer value; use d3d12 or vulkan");
            valid = false;
            break;
        }
    }
    LocalFree(argv);
    if (!valid) return nullptr;
    if (choice == Choice::Vulkan) {
        LOGI("renderer: selected vulkan");
        return CreateVulkanBackend();
    }
    LOGI("renderer: selected d3d12%s", specified ? "" : " (default)");
    return CreateD3D12Backend();
}

}  // namespace gpu
