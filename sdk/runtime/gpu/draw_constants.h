#pragma once

#include "gpu/shaders.h"

#include <cstring>
#include <cmath>

namespace gpu::draw_constants {
namespace detail {

inline float FloatBits(uint32_t value) {
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

inline void Color(uint32_t argb, float* out) {
    out[0] = float((argb >> 16) & 255) / 255.0f;
    out[1] = float((argb >> 8) & 255) / 255.0f;
    out[2] = float(argb & 255) / 255.0f;
    out[3] = float(argb >> 24) / 255.0f;
}

inline uint32_t Method(const State& state, uint32_t method) {
    return state.simple[method >> 2];
}

inline float Comparison(uint32_t value) {
    return float(value >= 0x200 && value <= 0x207 ? value - 0x200 : 7);
}

}  // namespace detail

inline void Fog(const State& state, bool vertex_program, DrawConstants& out) {
    if (!state.rs[kRsFogEnable]) {
        // Keep disabled fog distinct from direct table mode 0.
        out.fog[3] = 4.0f;
        return;
    }
    // With no table fog, a vertex program's oFog is the blend factor.
    if (vertex_program && state.rs[kRsFogTableMode] == 0) {
        out.fog[0] = 1.0f;
        out.fog[1] = 0.0f;
        out.fog[2] = 0.0f;
        out.fog[3] = 3.0f;
        return;
    }
    out.fog[0] = detail::FloatBits(state.rs[kRsFogStart]);
    out.fog[1] = detail::FloatBits(state.rs[kRsFogEnd]);
    out.fog[2] = detail::FloatBits(state.rs[kRsFogDensity]);
    out.fog[3] = float(state.rs[kRsFogTableMode]);
}

inline void Surface(const State& state, uint32_t width, uint32_t height, DrawConstants& out) {
    out.surf[0] = float(width);
    out.surf[1] = float(height);
    out.surf[2] = 1.0f / state.zscale;
    detail::Color(state.rs[kRsFogColor], out.fog_color);
    // D3DRS_RANGEFOGENABLE is vertex fog state. Fog color alpha is ignored by D3D;
    // carry the state bit here without changing the draw-constant layout.
    out.fog_color[3] = state.rs[kRsRangeFogEnable] ? 1.0f : 0.0f;
    out.misc[0] = float(detail::Method(state, kMAlphaRef) & 0xFF);
    out.misc[1] = detail::Method(state, kMAlphaTestEnable)
                      ? detail::Comparison(detail::Method(state, kMAlphaFunc)) : 7.0f;
    out.misc[2] = detail::Comparison(state.rs[kRsShadowFunc]);
    out.misc[3] = state.rs[kRsSpecularEnable] ? 1.0f : 0.0f;
    if (out.layout[0] == 0) out.layout[0] = 1;
    if (out.layout_y[0] == 0) out.layout_y[0] = 1;
}

// Pretransformed source: 0=RHW, 1=Z, 2=specular alpha.
inline uint32_t PretransformedFogSource(const State& state) {
    if (state.rs[kRsFogTableMode] == 0) return 2;  // specular alpha
    const auto& projection = state.transforms[3];
    return projection[3] == 0.0f && projection[7] == 0.0f && projection[11] == 0.0f && projection[15] == 1.0f ? 1 : 0;
}

inline void PretransformedFog(const State& state, bool pretransformed, DrawConstants& out) {
    if (pretransformed) out.fog_color[3] = float(PretransformedFogSource(state));
}

inline void Combiners(const State& state, DrawConstants& out) {
    for (int stage = 0; stage < 8; ++stage) {
        detail::Color(detail::Method(state, kMPsConstant0 + stage * 4), out.pc[stage * 2]);
        detail::Color(detail::Method(state, kMPsConstant1 + stage * 4), out.pc[stage * 2 + 1]);
    }
    detail::Color(detail::Method(state, kMPsFinalConstant0), out.pc[16]);
    detail::Color(detail::Method(state, kMPsFinalConstant1), out.pc[17]);
    detail::Color(state.rs[kRsTextureFactor], out.tfactor);
}

// D3DLIGHT8 as floats: Type, Diffuse[4], Specular[4], Ambient[4], Position[3], Direction[3], Range, Falloff,
// Attenuation0 to 2, Theta, Phi
constexpr int kLightFloats = 26;
constexpr int kLightDiffuse = 1;
constexpr int kLightSpecular = 5;
constexpr int kLightAmbient = 9;
constexpr int kLightPosition = 13;
constexpr int kLightDirection = 16;
constexpr int kLightRange = 19;
constexpr int kLightFalloff = 20;
constexpr int kLightAttenuation0 = 21;
constexpr int kLightAttenuation1 = 22;
constexpr int kLightAttenuation2 = 23;
constexpr int kLightTheta = 24;
constexpr int kLightPhi = 25;

inline void LightToView(const std::array<uint32_t, kLightFloats>& l, const float* vm, float (&o)[7][4]) {
    float f[kLightFloats];
    std::memcpy(f, l.data(), sizeof(f));
    for (int j = 0; j < 4; j++) {
        o[0][j] = f[kLightDiffuse + j];
        o[1][j] = f[kLightSpecular + j];
        o[2][j] = f[kLightAmbient + j];
    }
    const float* pos = f + kLightPosition;
    const float* dir = f + kLightDirection;
    for (int j = 0; j < 3; j++) {
        o[3][j] = pos[0] * vm[j] + pos[1] * vm[4 + j] + pos[2] * vm[8 + j] + vm[12 + j];
        o[4][j] = dir[0] * vm[j] + dir[1] * vm[4 + j] + dir[2] * vm[8 + j];
    }
    o[3][3] = f[kLightRange];
    o[4][3] = f[kLightFalloff];
    o[5][0] = f[kLightAttenuation0];
    o[5][1] = f[kLightAttenuation1];
    o[5][2] = f[kLightAttenuation2];
    o[6][0] = std::cos(f[kLightTheta] * 0.5f);
    o[6][1] = std::cos(f[kLightPhi] * 0.5f);
}

inline void Lights(const State& s, const FixedKey& fk, DrawConstants& c) {
    std::memcpy(c.mat, s.material.data(), sizeof(float) * 16);
    c.mat[4][0] = s.material[16];
    detail::Color(s.rs[kRsAmbient], c.amb);
    // lights to view space
    const float* vm = s.transforms[2].data();
    for (int i = 0; i < 8; i++) {
        if (!((fk.types >> (i * 2)) & 3)) continue;
        LightToView(s.lights[size_t(i)], vm, c.lt[i]);
    }
}

}  // namespace gpu::draw_constants
