#include "gpu/backend.h"

#include "gpu/remaster.h"

#include <filesystem>
#include <mutex>
#include <utility>

namespace gpu {

thread_local UiServices* g_ui_services = nullptr;
std::mutex g_override_lock;
std::vector<TextureOverride> g_override_set;
bool g_override_new = false;

UiScope::UiScope(UiServices* services) : previous_(g_ui_services) {
    g_ui_services = services;
}

UiScope::~UiScope() {
    g_ui_services = previous_;
}

uint64_t UiTexture(const std::string& pack_path) {
    if (!g_ui_services) return 0;
    std::filesystem::path path = std::filesystem::path(remaster::Dir()) / pack_path;
    remaster::Image image;
    const bool exists = std::filesystem::exists(path);
    const bool loaded = exists && remaster::ReadDds(path.wstring(), image);
    return g_ui_services->UiImage("pack:" + pack_path, loaded ? &image : nullptr, !exists);
}

uint64_t UiImage(const std::string& name, uint32_t width, uint32_t height, const uint8_t* rgba) {
    if (!g_ui_services || !rgba || !width || !height) return 0;
    remaster::Image image;
    image.fmt = DXGI_FORMAT_R8G8B8A8_UNORM;
    image.w = width;
    image.h = height;
    image.mips = 1;
    image.offsets = {0};
    image.data.assign(rgba, rgba + size_t(width) * height * 4);
    return g_ui_services->UiImage("img:" + name, &image, false);
}

bool TvScreen(float rect[4]) {
    return g_ui_services && g_ui_services->TvScreen(rect);
}

void SetTextureOverrides(std::vector<TextureOverride> set) {
    std::lock_guard lock(g_override_lock);
    g_override_set = std::move(set);
    g_override_new = true;
}

bool TakeTextureOverrides(std::vector<TextureOverride>& set) {
    std::lock_guard lock(g_override_lock);
    if (!g_override_new) return false;
    g_override_new = false;
    set = std::move(g_override_set);
    return true;
}

}  // namespace gpu
