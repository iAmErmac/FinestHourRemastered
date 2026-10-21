#pragma once

#include <cstdint>

namespace gpu::vulkan {

// Keep the final guest-image composition shader and its UBO layout in one
// product-owned ABI so offscreen checks exercise the same path as the backend.
inline constexpr char kBlitVertexShader[] = R"(
cbuffer B : register(b1) { float4 uvrect; float4 opts; };
struct O { float4 p : SV_Position; float2 uv : TEXCOORD0; };
O main(uint id : SV_VertexID) {
    O o; float2 t = float2((id << 1) & 2, id & 2);
    o.p = float4(t * float2(2, -2) + float2(-1, 1), 0, 1); o.uv = uvrect.xy + t * uvrect.zw; return o;
})";

inline constexpr char kBlitFragmentShader[] = R"(
cbuffer B : register(b1) { float4 uvrect; float4 opts; float4 gamma[192]; };
Texture2D img : register(t0); SamplerState smp : register(s0);
struct O { float4 p : SV_Position; float2 uv : TEXCOORD0; };
float GammaValue(float value, uint channel) {
    // D3D12 samples its 256-entry LUT with the point-filtered s1 sampler.
    // Normalized point sampling selects floor(coord * width), including the
    // exact 8-bit values and fractional colors produced by linear source filtering.
    uint index = min((uint)(saturate(value) * 256.0), 255u);
    float4 packed = gamma[channel * 64u + index / 4u];
    return packed[index & 3u];
}
float4 main(O i) : SV_Target {
    float4 color = img.Sample(smp, i.uv) * opts.x;
    if (opts.y > 0.5) color.rgb = float3(GammaValue(color.r, 0), GammaValue(color.g, 1), GammaValue(color.b, 2));
    return float4(color.rgb, 1.0);
}
)";

struct BlitConstants {
    float uv[4];
    float opts[4];
    float gamma[192][4];
};

}  // namespace gpu::vulkan
