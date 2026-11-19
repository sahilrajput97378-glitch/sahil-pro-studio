import io
import os
import tempfile
import librosa
import numpy as np
import soundfile as sf
import streamlit as st
from pedalboard import (
    Compressor,
    Delay,
    Gain,
    HighpassFilter,
    Limiter,
    Pedalboard,
    Reverb,
)

st.set_page_config(page_title="AI Vocal + Beat Reference Studio", page_icon="🎛️", layout="wide")
st.title("🎛️ AI Vocal + Beat Reference Studio")
st.caption("Vocal + Beat + Reference Song: AI रेफरेंस का अंदाज़ा लगाकर मिक्स, रिवर्ब, डिले और वाइब मैच करता है।")

OUT_DIR = "generated_songs"
os.makedirs(OUT_DIR, exist_ok=True)


def sanitize_audio(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def load_audio_file(uploaded_file, sr: int = 44100) -> np.ndarray:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        y, _ = librosa.load(tmp_path, sr=sr, mono=True)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return sanitize_audio(y)


def rms_db(y: np.ndarray) -> float:
    y = sanitize_audio(y)
    if len(y) == 0:
        return -60.0
    rms = float(np.sqrt(np.mean(np.square(y))))
    if rms < 1e-7:
        return -60.0
    return float(20.0 * np.log10(rms))


def analyze_reference_song(ref: np.ndarray, sr: int) -> dict:
    """Reference song ki vibe, tail/reverb, echo/delay aur dynamics ka safe andaaza."""
    ref = sanitize_audio(ref)
    defaults = {
        "reverb_wet": 0.18,
        "reverb_room": 0.42,
        "delay_mix": 0.06,
        "delay_feedback": 0.08,
        "comp_thresh": -18.0,
        "comp_ratio": 2.8,
        "beat_gain_db": -4.0,
        "duck_db": 1.8,
        "master_target_db": -12.0,
    }
    if len(ref) < sr:
        return defaults

    # 1. Dynamics / Crest factor -> Compression & Vibe
    rms_frames = librosa.feature.rms(y=ref, frame_length=2048, hop_length=512)[0]
    active_rms = rms_frames[rms_frames > np.max(rms_frames) * 0.15] if len(rms_frames) > 0 else np.array([0.05])
    mean_rms = float(np.mean(active_rms)) if len(active_rms) > 0 else 0.05
    peak = float(np.max(np.abs(ref))) + 1e-7
    crest_db = float(20.0 * np.log10(peak / max(mean_rms, 1e-6)))

    # Tighter/punchier modern reference -> slightly higher ratio
    comp_ratio = float(np.clip(2.0 + (18.0 - crest_db) * 0.14, 1.8, 4.5))
    comp_thresh = float(np.clip(-22.0 + (14.0 - crest_db) * 0.4, -26.0, -13.0))

    # 2. Reverb estimation from energy decay in low-energy frames
    low_mask = rms_frames < np.percentile(rms_frames, 35)
    low_energy = float(np.mean(rms_frames[low_mask])) if np.any(low_mask) else 0.0
    tail_ratio = float(np.clip(low_energy / max(mean_rms, 1e-6), 0.0, 1.0))
    reverb_wet = float(np.clip(0.10 + tail_ratio * 0.45, 0.08, 0.36))
    reverb_room = float(np.clip(0.30 + tail_ratio * 0.75, 0.25, 0.72))

    # 3. Safe Delay estimation (kept low-bounce so it never turns into glitchy echo)
    onset_env = librosa.onset.onset_strength(y=ref, sr=sr)
    if len(onset_env) > 64:
        autocorr = librosa.autocorrelate(onset_env, max_size=min(len(onset_env) - 1, 120))
        # Look around ~0.12s - 0.35s lag (approx 5 to 15 frames at hop=512, sr=44100)
        window = autocorr[5:16]
        if len(window) > 0 and np.max(autocorr) > 0:
            echo_score = float(np.clip(np.max(window) / (np.max(autocorr) + 1e-6), 0.0, 1.0))
        else:
            echo_score = 0.0
    else:
        echo_score = 0.0

    delay_mix = float(np.clip(0.03 + echo_score * 0.10, 0.02, 0.14))
    delay_feedback = float(np.clip(0.05 + echo_score * 0.10, 0.04, 0.16))

    # 4. Master loudness target (safe, non-clipping)
    ref_db = rms_db(ref)
    master_target_db = float(np.clip(ref_db, -18.0, -9.0))

    return {
        "reverb_wet": round(reverb_wet, 2),
        "reverb_room": round(reverb_room, 2),
        "delay_mix": round(delay_mix, 2),
        "delay_feedback": round(delay_feedback, 2),
        "comp_thresh": round(comp_thresh, 1),
        "comp_ratio": round(comp_ratio, 1),
        "beat_gain_db": -4.0,
        "duck_db": 1.8,
        "master_target_db": round(master_target_db, 1),
    }


def match_length_and_sr(vocal: np.ndarray, beat: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    vocal = sanitize_audio(vocal)
    beat = sanitize_audio(beat)
    target_len = max(len(vocal), len(beat))
    if target_len == 0:
        return np.zeros(sr, dtype=np.float32), np.zeros(sr, dtype=np.float32)

    vocal = np.pad(vocal, (0, max(0, target_len - len(vocal))), mode="constant")[:target_len]
    beat = np.pad(beat, (0, max(0, target_len - len(beat))), mode="constant")[:target_len]
    return vocal, beat


def process_vocal(
    vocal: np.ndarray,
    sr: int,
    hp_freq: float,
    comp_thresh: float,
    comp_ratio: float,
    delay_mix: float,
    delay_feedback: float,
    reverb_wet: float,
    reverb_room: float,
    vocal_gain_db: float,
) -> np.ndarray:
    vocal = sanitize_audio(vocal)
    if len(vocal) == 0 or np.max(np.abs(vocal)) < 1e-6:
        return vocal

    try:
        import noisereduce as nr
        vocal = nr.reduce_noise(y=vocal, sr=sr, prop_decrease=0.45, stationary=True)
        vocal = sanitize_audio(vocal)
    except Exception:
        pass

    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=hp_freq),
        Compressor(threshold_db=comp_thresh, ratio=comp_ratio, attack_ms=8.0, release_ms=120.0),
        Delay(delay_seconds=0.18, feedback=delay_feedback, mix=delay_mix),
        Reverb(room_size=reverb_room, damping=0.55, wet_level=reverb_wet, dry_level=0.85, width=0.8),
        Gain(gain_db=vocal_gain_db),
        Limiter(threshold_db=-1.0, release_ms=100.0),
    ])
    return sanitize_audio(board(vocal, sr))


def apply_gentle_ducking(vocal: np.ndarray, beat: np.ndarray, sr: int, duck_db: float) -> np.ndarray:
    vocal, beat = match_length_and_sr(vocal, beat, sr)
    if len(vocal) == 0 or duck_db <= 0.0:
        return beat

    frame_len = max(256, int(0.03 * sr))
    hop_len = max(128, int(0.015 * sr))
    rms = librosa.feature.rms(y=vocal, frame_length=frame_len, hop_length=hop_len)[0]
    if len(rms) == 0:
        return beat

    peak_rms = max(float(np.max(rms)), 1e-5)
    norm_env = np.clip(rms / peak_rms, 0.0, 1.0)
    active_mask = np.clip((norm_env - 0.12) / 0.65, 0.0, 1.0)

    gain_db_curve = -duck_db * active_mask
    gain_lin_curve = librosa.db_to_amplitude(gain_db_curve)

    times = librosa.frames_to_time(np.arange(len(gain_lin_curve)), sr=sr, hop_length=hop_len)
    sample_times = np.arange(len(beat)) / float(sr)
    beat_gain = np.interp(sample_times, times, gain_lin_curve, left=1.0, right=1.0).astype(np.float32)

    return sanitize_audio(beat * beat_gain)


def mix_and_master(
    vocal: np.ndarray,
    beat: np.ndarray,
    sr: int,
    beat_gain_db: float,
    vocal_offset_sec: float,
    duck_db: float,
    master_target_db: float,
) -> np.ndarray:
    vocal = sanitize_audio(vocal)
    beat = sanitize_audio(beat)

    shift_samples = int(round(vocal_offset_sec * sr))
    if shift_samples > 0:
        vocal = np.concatenate([np.zeros(shift_samples, dtype=np.float32), vocal])
    elif shift_samples < 0:
        vocal = vocal[min(abs(shift_samples), len(vocal)):]

    vocal, beat = match_length_and_sr(vocal, beat, sr)
    beat_adj = beat * float(librosa.db_to_amplitude(beat_gain_db))
    beat_ducked = apply_gentle_ducking(vocal, beat_adj, sr, duck_db=duck_db)
    combined = sanitize_audio(vocal + beat_ducked)

    current_db = rms_db(combined)
    if current_db > -55.0:
        level_shift_db = float(np.clip(master_target_db - current_db, -8.0, 8.0))
        combined = sanitize_audio(combined * float(librosa.db_to_amplitude(level_shift_db)))

    master_board = Pedalboard([
        Compressor(threshold_db=-10.0, ratio=2.2, attack_ms=15.0, release_ms=180.0),
        Limiter(threshold_db=-0.6, release_ms=120.0),
    ])
    return sanitize_audio(master_board(combined, sr))


# Session state for AI Reference Brain
if "ai_params" not in st.session_state:
    st.session_state.ai_params = {
        "reverb_wet": 0.18,
        "reverb_room": 0.42,
        "delay_mix": 0.06,
        "delay_feedback": 0.08,
        "comp_thresh": -18.0,
        "comp_ratio": 2.8,
        "beat_gain_db": -4.0,
        "duck_db": 1.8,
        "master_target_db": -12.0,
    }

col1, col2, col3 = st.columns(3)
with col1:
    vocal_file = st.file_uploader("🎤 Apni Vocal Track", type=["wav", "mp3", "ogg", "m4a"])
with col2:
    beat_file = st.file_uploader("🥁 Beat / Instrumental Track", type=["wav", "mp3", "ogg", "m4a"])
with col3:
    ref_file = st.file_uploader("🎧 Reference Song (Vibe & Mix Match)", type=["wav", "mp3", "ogg", "m4a"])

ai_btn_col, info_col = st.columns([1, 2])
with ai_btn_col:
    run_ai_brain = st.button("🧠 AI Dimag: Reference Analyze Karo", use_container_width=True)

if run_ai_brain:
    if ref_file is None:
        st.warning("Pehle Reference Song upload karo, phir AI Dimag usko analyze karega.")
    else:
        with st.spinner("AI Reference Song ki vibe, reverb, delay aur mastering level samajh raha hai..."):
            ref_audio = load_audio_file(ref_file, sr=44100)
            if len(ref_audio) == 0:
                st.error("Reference file khaali lag rahi hai.")
            else:
                st.session_state.ai_params = analyze_reference_song(ref_audio, sr=44100)
                st.success("AI ne reference song analyze kar liya aur settings set kar di hain!")

p = st.session_state.ai_params

with st.sidebar:
    st.header("🎚️ Vocal & Mix Controls")
    hp_freq = st.slider("Highpass Filter (Hz)", 50.0, 200.0, 95.0, 5.0)
    comp_thresh = st.slider("Vocal Compressor Threshold (dB)", -35.0, -5.0, float(p["comp_thresh"]), 0.5)
    comp_ratio = st.slider("Vocal Compressor Ratio", 1.5, 6.0, float(p["comp_ratio"]), 0.1)
    vocal_gain_db = st.slider("Vocal Level (dB)", -12.0, 12.0, 2.0, 0.5)

    st.divider()
    st.subheader("Reverb & Delay (AI Matched)")
    reverb_wet = st.slider("Reverb Wet", 0.0, 0.5, float(p["reverb_wet"]), 0.01)
    reverb_room = st.slider("Reverb Room Size", 0.1, 0.9, float(p["reverb_room"]), 0.02)
    delay_mix = st.slider("Delay Mix", 0.0, 0.3, float(p["delay_mix"]), 0.01)
    delay_feedback = st.slider("Delay Feedback", 0.0, 0.35, float(p["delay_feedback"]), 0.01)

    st.divider()
    st.subheader("Beat & Mastering")
    beat_gain_db = st.slider("Beat Volume (dB)", -18.0, 6.0, float(p["beat_gain_db"]), 0.5)
    duck_db = st.slider("Auto Beat Ducking (dB)", 0.0, 5.0, float(p["duck_db"]), 0.2)
    master_target_db = st.slider("Master Loudness Target (dB)", -20.0, -7.0, float(p["master_target_db"]), 0.5)
    vocal_offset_sec = st.slider("Vocal Timing Shift (sec)", -2.0, 2.0, 0.0, 0.05)

with info_col:
    st.info(
        f"AI Preset — Reverb: {reverb_wet:.2f} | Delay: {delay_mix:.2f} | "
        f"Comp: {comp_ratio:.1f}:1 ({comp_thresh:.1f} dB) | Master: {master_target_db:.1f} dB"
    )

track_name = st.text_input("Output Song Name", value="AI_Reference_Mix")

if st.button("✨ Reference Ke Hisaab Se Mix & Master Karo", type="primary", use_container_width=True):
    if vocal_file is None or beat_file is None:
        st.warning("Pehle apni Vocal aur Beat dono files upload karo.")
    else:
        with st.spinner("Vocal aur Beat ko reference vibe ke hisaab se mix aur master kiya ja raha hai..."):
            sr = 44100
            # Agar reference upload hai aur user ne alag se AI button nahi dabaya toh bhi ek baar auto-analyze kar lo
            if ref_file is not None and not run_ai_brain:
                ref_audio = load_audio_file(ref_file, sr=sr)
                if len(ref_audio) > 0:
                    auto_p = analyze_reference_song(ref_audio, sr=sr)
                    reverb_wet = auto_p["reverb_wet"]
                    reverb_room = auto_p["reverb_room"]
                    delay_mix = auto_p["delay_mix"]
                    delay_feedback = auto_p["delay_feedback"]
                    comp_thresh = auto_p["comp_thresh"]
                    comp_ratio = auto_p["comp_ratio"]
                    master_target_db = auto_p["master_target_db"]

            vocal_raw = load_audio_file(vocal_file, sr=sr)
            beat_raw = load_audio_file(beat_file, sr=sr)

            if len(vocal_raw) == 0 or len(beat_raw) == 0:
                st.error("Vocal ya Beat file khaali hai.")
            else:
                vocal_clean = process_vocal(
                    vocal_raw,
                    sr=sr,
                    hp_freq=hp_freq,
                    comp_thresh=comp_thresh,
                    comp_ratio=comp_ratio,
                    delay_mix=delay_mix,
                    delay_feedback=delay_feedback,
                    reverb_wet=reverb_wet,
                    reverb_room=reverb_room,
                    vocal_gain_db=vocal_gain_db,
                )

                final_mix = mix_and_master(
                    vocal=vocal_clean,
                    beat=beat_raw,
                    sr=sr,
                    beat_gain_db=beat_gain_db,
                    vocal_offset_sec=vocal_offset_sec,
                    duck_db=duck_db,
                    master_target_db=master_target_db,
                )

                if np.max(np.abs(final_mix)) < 1e-6:
                    st.error("Output audio silent aaya—input levels check karo.")
                else:
                    safe_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in track_name.strip()) or "Mix"
                    out_path = os.path.join(OUT_DIR, f"{safe_name}_Master.wav")
                    sf.write(out_path, final_mix, sr)

                    buf = io.BytesIO()
                    sf.write(buf, final_mix, sr, format="WAV")
                    buf.seek(0)

                    st.success("Gaana mix aur master hokar taiyaar hai!")
                    st.audio(buf.getvalue(), format="audio/wav")
                    st.download_button(
                        label="⬇️ Download Final Song (WAV)",
                        data=buf.getvalue(),
                        file_name=f"{safe_name}_Master.wav",
                        mime="audio/wav",
                        use_container_width=True,
                    )
