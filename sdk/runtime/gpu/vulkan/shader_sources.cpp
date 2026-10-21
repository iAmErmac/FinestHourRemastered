#include "gpu/vulkan/shader_sources.h"

#include <functional>
#include <utility>

namespace gpu::vulkan {
namespace {

constexpr uint64_t kSeed = 1469598103934665603ull;

void Add(size_t& hash, uint64_t value) {
    hash ^= std::hash<uint64_t>{}(value) + size_t(0x9e3779b9) + (hash << 6) + (hash >> 2);
}

void AddPixel(size_t& hash, const PixelKey& pixel) {
    Add(hash, pixel.programmable);
    Add(hash, pixel.count);
    for (uint32_t value : pixel.rgb_in) Add(hash, value);
    for (uint32_t value : pixel.rgb_out) Add(hash, value);
    for (uint32_t value : pixel.a_in) Add(hash, value);
    for (uint32_t value : pixel.a_out) Add(hash, value);
    Add(hash, pixel.final_abcd);
    Add(hash, pixel.final_efg);
    Add(hash, pixel.tex_modes);
    Add(hash, pixel.dot_mapping);
    Add(hash, pixel.input_texture);
    Add(hash, pixel.compare_mode);
    for (const auto& stage : pixel.ff)
        for (uint32_t value : stage) Add(hash, value);
    for (const TexInfo& texture : pixel.tex) {
        Add(hash, texture.kind);
        Add(hash, texture.rect);
        Add(hash, texture.shadow);
        Add(hash, texture.flags);
    }
    Add(hash, pixel.alphakill);
    Add(hash, pixel.flat);
    Add(hash, pixel.two_sided);
    Add(hash, pixel.proj);
    Add(hash, pixel.quad);
    Add(hash, pixel.lit);
    Add(hash, pixel.soft);
    Add(hash, pixel.gbuf);
}

}  // namespace

size_t ShaderSourceRecipeHash::operator()(const ShaderSourceRecipe& recipe) const {
    size_t hash = kSeed;
    Add(hash, recipe.program_hash);
    Add(hash, recipe.program_words);
    Add(hash, recipe.declaration_handle);
    Add(hash, recipe.regs);
    Add(hash, recipe.decode);
    Add(hash, recipe.programmable);
    Add(hash, recipe.pretransformed);
    Add(hash, recipe.point_geometry);
    Add(hash, recipe.fixed.light);
    Add(hash, recipe.fixed.types);
    for (uint32_t value : recipe.fixed.tex) Add(hash, value);
    AddPixel(hash, recipe.pixel);
    Add(hash, reinterpret_cast<uintptr_t>(recipe.hooks));
    return hash;
}

const DrawSources& DrawSourceCache::Get(VertexShader* vertex, uint32_t vs_handle, uint16_t regs, uint64_t decode,
                                        bool pretransformed, const FixedKey& fixed, const PixelKey& pixel,
                                        bool point_geometry, bool* cache_hit) {
    const bool programmable = (vs_handle & 1) && vertex && !vertex->program.empty();
    uint64_t program_hash = 0;
    uint32_t program_words = 0;
    if (programmable) {
        if (!vertex->hash) vertex->hash = Hash(vertex->program.data(), vertex->program.size() * sizeof(uint32_t)) | 1;
        program_hash = vertex->hash;
        program_words = static_cast<uint32_t>(vertex->program.size());
    }
    ShaderSourceRecipe recipe;
    recipe.program_hash = program_hash;
    recipe.program_words = program_words;
    recipe.declaration_handle = programmable ? 0 : vs_handle;
    recipe.regs = regs;
    recipe.decode = decode;
    recipe.programmable = programmable;
    recipe.pretransformed = pretransformed;
    recipe.point_geometry = point_geometry;
    recipe.fixed = fixed;
    recipe.pixel = pixel;
    recipe.hooks = g_shader_hooks;
    if (auto found = sources_.find(recipe); found != sources_.end()) {
        if (cache_hit) *cache_hit = true;
        return found->second;
    }
    if (cache_hit) *cache_hit = false;
    DrawSources sources;
    std::string base = programmable ? VertexProgramHlsl(vertex->program, regs, decode)
                                    : FixedVertexHlsl(regs, pretransformed, decode, fixed);
    sources.vertex = pixel.quad ? QuadHlsl(std::move(base)) : std::move(base);
    sources.pixel = PixelHlsl(pixel);
    if (point_geometry) sources.geometry = PointSpriteGsHlsl();
    return sources_.emplace(std::move(recipe), std::move(sources)).first->second;
}

void DrawSourceCache::Clear() { sources_.clear(); }

}  // namespace gpu::vulkan
