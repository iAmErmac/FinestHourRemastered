#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "gpu/state.h"

namespace gpu {

// What the renderer uploads a console texture format as
enum class Host : uint8_t { BGRA8, BC1, BC2, BC3, R32F, R16Unorm, None };

struct FormatInfo {
    const char* name;
    uint8_t bits;      // per texel, 4 or 8 bytes per block for the DXT formats
    bool swizzled;     // Morton order in memory
    bool compressed;   // 4x4 blocks
    bool depth;
    Host host;
};

const FormatInfo& Format(uint32_t xbox_format);

// a D3DPixelContainer decoded
struct TextureDesc {
    uint32_t data = 0;  // guest virtual address of level 0
    uint32_t format = 0;
    uint32_t width = 1, height = 1, depth = 1, levels = 1;
    uint32_t pitch = 0;  // linear formats only
    bool cube = false;
    bool linear = false;
};

// The active P8 palette, including the XDK size-code contract and a cached content hash.
struct PaletteInfo {
    const uint32_t* colours = nullptr;
    uint32_t va = 0;
    size_t bytes = 0;
    uint64_t hash = 0;
};

void ReadPalette(const State& state, int stage, const TextureDesc& desc, PaletteInfo& palette);

bool Describe(uint32_t header, TextureDesc& out);
bool DescribeSurface(const Surface& s, TextureDesc& out);
uint32_t LevelSize(const TextureDesc& d, uint32_t level);
uint32_t LevelOffset(const TextureDesc& d, uint32_t level);
uint32_t FaceSize(const TextureDesc& d);  // cube face stride
uint32_t TotalSize(const TextureDesc& d);

// A cheap per-frame guest-content check: small resources are hashed whole, larger ones are sampled evenly.
uint64_t ContentSignature(const TextureDesc& d);

// converts one level (one face) from guest memory into what Host describes, rows tightly packed
void DecodeLevel(const TextureDesc& d, uint32_t level, uint32_t face, const uint8_t* guest, const uint32_t* palette,
                 std::vector<uint8_t>& out, uint32_t& row_pitch);

// Decodes one uncompressed guest texel into the renderer's B, G, R, A order.
void DecodePixel(uint32_t format, const uint8_t* guest, uint8_t* host);

// Supported synchronous render-target writeback storage and the shared scaled encoder used by Vulkan.
bool CanWriteBackSurface(const TextureDesc& d);
bool WriteBackSurface(const TextureDesc& d, const uint8_t* source, uint32_t source_width, uint32_t source_height,
                      uint32_t source_pitch, size_t source_bytes, uint8_t* guest, size_t guest_bytes,
                      std::vector<uint8_t>* canonical);

uint32_t LinearColorFormat(uint32_t fmt);
uint32_t LinearDepthFormat(uint32_t fmt);
void SurfaceSize(const Surface& s, uint32_t& w, uint32_t& h);

// CopyRects done in guest memory, false when the formats do not allow it
bool CopySurfaceMemory(const Surface& src, const Surface& dst, uint32_t count, uint32_t rects, uint32_t points);

// the title locked a resource, the next bind re-uploads it
void MarkDirty(uint32_t resource);
uint64_t DirtyEpoch(uint32_t data_va);

// vertex declarations and pixel shader constant routing, both follow the XDK's definitions
void ParseDeclaration(VertexShader& vs);
// SetPixelShaderConstant: the stages of the current shader mapped to these registers take the values
void ApplyPixelConstants(State& s, uint32_t first, uint32_t count);
uint32_t PackColor(const std::array<float, 4>& c);

}  // namespace gpu
