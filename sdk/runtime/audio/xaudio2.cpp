// mixer output through XAudio2
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <xaudio2.h>

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "audio/audio.h"
#include "core/log.h"

namespace audio {

namespace {
std::mutex g_tap_lock;
std::function<void(const float*, uint32_t)> g_tap;
std::once_flag g_mute_log_once;

bool AudioMuteRequested() {
    const char* value = std::getenv("XBE_AUDIO_MUTE");
    return value && value[0] && std::strcmp(value, "0") != 0;
}

bool ApplyOutputMute(IXAudio2MasteringVoice* master, std::string& error) {
    if (!AudioMuteRequested()) return true;
    const HRESULT hr = master->SetVolume(0.0f);
    if (FAILED(hr)) {
        error = "IXAudio2MasteringVoice::SetVolume(0) failed (" + std::to_string(unsigned(hr)) + ")";
        return false;
    }
    std::call_once(g_mute_log_once, [] {
        LOGR("audio: developer output mute active via XBE_AUDIO_MUTE; XAudio2 voices and render callbacks remain enabled");
    });
    return true;
}

void DestroyOutput(IXAudio2* xa, IXAudio2MasteringVoice* master, IXAudio2SourceVoice* voice, HRESULT com) {
    if (voice) voice->DestroyVoice();
    if (master) master->DestroyVoice();
    if (xa) xa->Release();
    if (SUCCEEDED(com)) CoUninitialize();
}
}  // namespace

void SetTap(std::function<void(const float*, uint32_t)> tap) {
    std::lock_guard l(g_tap_lock);
    g_tap = std::move(tap);
}

void StartOutput(std::function<void(float*, uint32_t)> render) {
    std::thread([render] {
        const HRESULT com = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        IXAudio2* xa = nullptr;
        IXAudio2MasteringVoice* master = nullptr;
        IXAudio2SourceVoice* voice = nullptr;
        WAVEFORMATEX wf{};
        wf.wFormatTag = WAVE_FORMAT_IEEE_FLOAT;
        wf.nChannels = 2;
        wf.nSamplesPerSec = kRate;
        wf.wBitsPerSample = 32;
        wf.nBlockAlign = 8;
        wf.nAvgBytesPerSec = kRate * 8;
        auto failed = [&](const char* operation, HRESULT hr) {
            if (SUCCEEDED(hr)) return false;
            LOGE("audio: %s failed (%08X), no sound", operation, unsigned(hr));
            DestroyOutput(xa, master, voice, com);
            return true;
        };
        if (failed("COM initialization", com) ||
            failed("XAudio2Create", XAudio2Create(&xa, 0, XAUDIO2_DEFAULT_PROCESSOR)) ||
            failed("CreateMasteringVoice", xa->CreateMasteringVoice(&master)) ||
            failed("CreateSourceVoice", xa->CreateSourceVoice(&voice, &wf)))
            return;
        std::string mute_error;
        if (!ApplyOutputMute(master, mute_error)) {
            LOGE("audio: %s, no sound", mute_error.c_str());
            DestroyOutput(xa, master, voice, com);
            return;
        }
        if (failed("StartSourceVoice", voice->Start())) return;
        constexpr uint32_t kFrames = kRate / 100, kSlots = 4;
        std::vector<float> bufs(kSlots * kFrames * 2);
        for (uint32_t n = 0;;) {
            XAUDIO2_VOICE_STATE st;
            voice->GetState(&st, XAUDIO2_VOICE_NOSAMPLESPLAYED);
            if (st.BuffersQueued >= 3) {
                Sleep(2);
                continue;
            }
            float* b = &bufs[(n % kSlots) * kFrames * 2];
            render(b, kFrames);
            {
                std::lock_guard l(g_tap_lock);
                if (g_tap) g_tap(b, kFrames);
            }
            XAUDIO2_BUFFER xb{};
            xb.AudioBytes = kFrames * 8;
            xb.pAudioData = reinterpret_cast<const BYTE*>(b);
            voice->SubmitSourceBuffer(&xb);
            n++;  // only a submitted buffer takes a slot, XAudio2 still reads the queued ones
        }
    }).detach();
}

}  // namespace audio
