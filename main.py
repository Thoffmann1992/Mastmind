from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, butter
import os
import shutil
import uuid
from pathlib import Path

app = FastAPI(title="MasterMind API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# ── EQ ──────────────────────────────────────────────────────────────
def make_shelf(freq, gain_db, shelf_type, sr):
    """Low/high shelf filter — second-order sections"""
    from scipy.signal import iirfilter
    nyq = sr / 2
    norm = freq / nyq
    norm = np.clip(norm, 1e-4, 0.99)
    b, a = butter(2, norm, btype='low' if shelf_type == 'low' else 'high')
    lin = 10 ** (gain_db / 20)
    if gain_db > 0:
        b = b * lin
    else:
        a_scaled = np.array(a)
        a_scaled[0] *= lin
        b = b / lin if lin != 0 else b
    return b, a

def apply_shelf(y, sr, freq, gain_db, shelf_type='low'):
    if abs(gain_db) < 0.1:
        return y
    nyq = sr / 2
    norm = np.clip(freq / nyq, 1e-4, 0.99)
    from scipy.signal import lfilter
    b, a = butter(2, norm, btype='low' if shelf_type == 'low' else 'high')
    gain_lin = 10 ** (gain_db / 20)
    filtered = lfilter(b, a, y, axis=0)
    return y + filtered * (gain_lin - 1)

def apply_peaking(y, sr, freq, gain_db, q=1.4):
    """Peaking EQ band"""
    if abs(gain_db) < 0.1:
        return y
    from scipy.signal import lfilter
    w0 = 2 * np.pi * freq / sr
    A = 10 ** (gain_db / 40)
    alpha = np.sin(w0) / (2 * q)
    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A
    b = np.array([b0/a0, b1/a0, b2/a0])
    a = np.array([1.0, a1/a0, a2/a0])
    return lfilter(b, a, y, axis=0)

def apply_highpass(y, sr, cutoff, order=2):
    from scipy.signal import lfilter
    nyq = sr / 2
    norm = np.clip(cutoff / nyq, 1e-4, 0.99)
    b, a = butter(order, norm, btype='high')
    return lfilter(b, a, y, axis=0)

# ── COMPRESSOR ───────────────────────────────────────────────────────
def compress(y, threshold_db=-18, ratio=4.0, attack_ms=10, release_ms=150, sr=44100, makeup_db=0):
    """RMS-based soft-knee compressor"""
    threshold = 10 ** (threshold_db / 20)
    makeup = 10 ** (makeup_db / 20)
    attack  = 1 - np.exp(-1 / (sr * attack_ms  / 1000))
    release = 1 - np.exp(-1 / (sr * release_ms / 1000))

    mono = y if y.ndim == 1 else np.mean(y, axis=1)
    gain = np.ones(len(mono))
    env = 0.0
    for i in range(len(mono)):
        level = abs(mono[i])
        if level > env:
            env = env + attack * (level - env)
        else:
            env = env + release * (level - env)
        if env > threshold:
            reduction = threshold * (env / threshold) ** (1 / ratio) / env
        else:
            reduction = 1.0
        gain[i] = reduction

    if y.ndim == 1:
        return y * gain * makeup
    else:
        return y * gain[:, np.newaxis] * makeup

# ── LIMITER ──────────────────────────────────────────────────────────
def limit(y, ceiling_db=-0.3):
    ceiling = 10 ** (ceiling_db / 20)
    peak = np.max(np.abs(y))
    if peak > ceiling:
        y = y * (ceiling / peak)
    return y

# ── STEREO WIDTH ─────────────────────────────────────────────────────
def stereo_width(y, width=1.2):
    if y.ndim == 1:
        y = np.stack([y, y], axis=1)
    mid  = (y[:, 0] + y[:, 1]) / 2
    side = (y[:, 0] - y[:, 1]) / 2 * width
    y[:, 0] = mid + side
    y[:, 1] = mid - side
    return y

# ── LUFS NORMALIZE ───────────────────────────────────────────────────
def normalize_lufs(y, target_lufs=-14.0):
    mono = y if y.ndim == 1 else np.mean(y, axis=1)
    rms = np.sqrt(np.mean(mono ** 2) + 1e-10)
    current_lufs = 20 * np.log10(rms) - 0.691
    gain_db = target_lufs - current_lufs
    gain_db = np.clip(gain_db, -20, 20)
    return y * (10 ** (gain_db / 20))

# ── NOISE GATE ───────────────────────────────────────────────────────
def noise_gate(y, sr, amount=0.3):
    if amount <= 0:
        return y
    mono = y if y.ndim == 1 else np.mean(y, axis=1)
    S = np.abs(librosa.stft(mono))
    noise_floor = np.percentile(S, 15) * (1 + amount * 2)
    mask = S > noise_floor
    # Smooth mask
    from scipy.ndimage import uniform_filter
    mask = uniform_filter(mask.astype(float), size=(1, 5)) > 0.5
    S_gated = S * mask
    phase = np.angle(librosa.stft(mono))
    y_clean = librosa.istft(S_gated * np.exp(1j * phase))
    # Match length
    if len(y_clean) < len(mono):
        y_clean = np.pad(y_clean, (0, len(mono) - len(y_clean)))
    else:
        y_clean = y_clean[:len(mono)]
    if y.ndim == 2:
        return np.stack([y_clean, y_clean], axis=1)
    return y_clean

# ── TAPE SATURATION (Warm profil üçün) ──────────────────────────────
def tape_saturate(y, amount=0.3):
    """Soft clip + harmonic distortion simulyasiyası"""
    drive = 1 + amount * 4
    y_driven = y * drive
    # Soft clipping: tanh
    saturated = np.tanh(y_driven) / np.tanh(drive)
    return saturated * (1 - amount * 0.1) + y * amount * 0.1

# ════════════════════════════════════════════════════════════════════
# 4 MASTERİNG PROFİLİ
# ════════════════════════════════════════════════════════════════════

def profile_streaming(y, sr):
    """
    Streaming (Spotify/Apple Music)
    Target: -14 LUFS, True Peak -1 dBTP
    Geniş dinamika, natural tezlik balansı
    """
    # HP filter — rumble təmizlə
    y = apply_highpass(y, sr, cutoff=30, order=2)
    # EQ
    y = apply_shelf(y, sr, freq=100, gain_db=1.5, shelf_type='low')       # Bass lift
    y = apply_peaking(y, sr, freq=300, gain_db=-1.0, q=1.2)               # Mud azalt
    y = apply_peaking(y, sr, freq=3000, gain_db=1.0, q=1.5)               # Presence
    y = apply_shelf(y, sr, freq=10000, gain_db=1.5, shelf_type='high')    # Air
    # Compression — yüngül
    y = compress(y, threshold_db=-20, ratio=3.0, attack_ms=15, release_ms=200, sr=sr, makeup_db=2)
    # Stereo
    y = stereo_width(y, width=1.1)
    # LUFS normalize
    y = normalize_lufs(y, target_lufs=-14.0)
    # Limit
    y = limit(y, ceiling_db=-1.0)
    return y

def profile_club(y, sr):
    """
    Club / Loud
    Target: -8 LUFS, maksimum enerji
    Güclü bas, sıx kompressor, punch
    """
    y = apply_highpass(y, sr, cutoff=20, order=2)
    # EQ — ağır bas artır
    y = apply_shelf(y, sr, freq=80, gain_db=4.0, shelf_type='low')        # Sub bass boost
    y = apply_peaking(y, sr, freq=120, gain_db=3.0, q=1.0)               # Kick punch
    y = apply_peaking(y, sr, freq=400, gain_db=-2.0, q=1.2)              # Mud cut
    y = apply_peaking(y, sr, freq=5000, gain_db=2.0, q=1.5)              # Presence/snap
    y = apply_shelf(y, sr, freq=12000, gain_db=2.0, shelf_type='high')   # Clarity
    # Compression — sıx, punch
    y = compress(y, threshold_db=-14, ratio=6.0, attack_ms=5, release_ms=80, sr=sr, makeup_db=4)
    # Multi-band style: bas ayrıca sıxılır
    y = stereo_width(y, width=1.3)
    y = normalize_lufs(y, target_lufs=-8.0)
    y = limit(y, ceiling_db=-0.3)
    return y

def profile_warm(y, sr):
    """
    Warm / Vintage
    Target: -12 LUFS, analog karakter
    Tape saturation, yumşaq tezlik, istilik
    """
    y = apply_highpass(y, sr, cutoff=40, order=2)
    # EQ — vintage karakter
    y = apply_shelf(y, sr, freq=120, gain_db=2.5, shelf_type='low')       # Warm bass
    y = apply_peaking(y, sr, freq=800, gain_db=2.0, q=1.0)               # Midrange richness
    y = apply_peaking(y, sr, freq=3500, gain_db=1.5, q=1.2)              # Vocal warmth
    y = apply_shelf(y, sr, freq=8000, gain_db=-2.0, shelf_type='high')   # Tiz azalt (analog)
    # Tape saturation
    y_mono = y if y.ndim == 1 else np.mean(y, axis=1)
    y = tape_saturate(y, amount=0.35)
    # Yüngül kompressor
    y = compress(y, threshold_db=-22, ratio=2.5, attack_ms=20, release_ms=300, sr=sr, makeup_db=3)
    # Stereo — təbii, dar deyil
    y = stereo_width(y, width=0.95)
    y = normalize_lufs(y, target_lufs=-12.0)
    y = limit(y, ceiling_db=-0.5)
    return y

def profile_cinematic(y, sr):
    """
    Cinematic / Film
    Target: -16 LUFS, geniş dinamika aralığı
    Geniş stereo, dramatik, sub-bass extension
    """
    y = apply_highpass(y, sr, cutoff=25, order=2)
    # EQ — geniş, açıq səs
    y = apply_shelf(y, sr, freq=60, gain_db=3.0, shelf_type='low')        # Sub extension
    y = apply_peaking(y, sr, freq=200, gain_db=-1.5, q=1.5)              # Mud azalt
    y = apply_peaking(y, sr, freq=1000, gain_db=-0.5, q=2.0)             # Nasal azalt
    y = apply_peaking(y, sr, freq=4000, gain_db=2.0, q=1.2)              # Detail
    y = apply_shelf(y, sr, freq=10000, gain_db=3.0, shelf_type='high')   # Air / openness
    # Yüngül kompressor — dinamika saxla
    y = compress(y, threshold_db=-24, ratio=2.0, attack_ms=25, release_ms=400, sr=sr, makeup_db=2)
    # Geniş stereo
    y = stereo_width(y, width=1.5)
    y = normalize_lufs(y, target_lufs=-16.0)
    y = limit(y, ceiling_db=-1.0)
    return y

PROFILES = {
    "streaming": profile_streaming,
    "club":      profile_club,
    "warm":      profile_warm,
    "cinematic": profile_cinematic,
}

# ── API ENDPOINTS ────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "profiles": list(PROFILES.keys())}

@app.post("/master")
async def master_audio(
    file: UploadFile = File(...),
    profile: str = Form("streaming"),
    noise_reduction: float = Form(0.0),
):
    # Validation
    if not file.content_type or not file.content_type.startswith("audio/"):
        # Try by extension
        ext = Path(file.filename or "").suffix.lower()
        if ext not in [".mp3", ".wav", ".flac", ".aiff", ".ogg", ".m4a"]:
            raise HTTPException(400, "Yalnız audio fayllar qəbul olunur")

    if profile not in PROFILES:
        raise HTTPException(400, f"Profil tapılmadı. Mövcud profillər: {list(PROFILES.keys())}")

    uid = str(uuid.uuid4())[:8]
    suffix = Path(file.filename or "input.mp3").suffix or ".mp3"
    input_path = UPLOAD_DIR / f"{uid}_input{suffix}"
    output_path = PROCESSED_DIR / f"{uid}_{profile}_master.wav"

    # Save upload
    try:
        with input_path.open("wb") as buf:
            shutil.copyfileobj(file.file, buf)
    except Exception as e:
        raise HTTPException(500, f"Fayl saxlanmadı: {e}")

    try:
        # Load audio
        y, sr = librosa.load(str(input_path), sr=None, mono=False)

        # librosa mono yükləyir — stereo lazımdır
        if y.ndim == 1:
            y = np.stack([y, y], axis=1)
        else:
            y = y.T  # (channels, samples) → (samples, channels)

        # Noise reduction
        if noise_reduction > 0:
            y = noise_gate(y, sr, amount=noise_reduction)

        # Apply profile
        y = PROFILES[profile](y, sr)

        # Final clip guard
        y = np.clip(y, -1.0, 1.0)

        # Write 24-bit WAV
        sf.write(str(output_path), y, sr, subtype="PCM_24")

        orig_name = Path(file.filename or "audio").stem
        dl_name = f"{orig_name}_{profile}_master.wav"

        return FileResponse(
            path=str(output_path),
            filename=dl_name,
            media_type="audio/wav",
            headers={"X-Profile": profile, "X-Sample-Rate": str(sr)},
            background=None,
        )

    except librosa.util.exceptions.ParameterError as e:
        raise HTTPException(422, f"Audio format xətası: {e}")
    except Exception as e:
        raise HTTPException(500, f"Mastering xətası: {e}")
    finally:
        if input_path.exists():
            input_path.unlink()

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse("""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>MasterMind API</title>
<style>body{font-family:monospace;background:#080810;color:#e2ff4e;padding:40px;}</style>
</head><body>
<h1>🎛️ MasterMind API</h1>
<p>POST /master — audio yüklə, master al</p>
<p>GET /health — status yoxla</p>
<p>Profillər: streaming · club · warm · cinematic</p>
</body></html>""")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
