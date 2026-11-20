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
    HighShelfFilter,
    Limiter,
    Pedalboard,
    Reverb,
)

st.set_page_config(page_title="Reference Vibe & Vocal Matcher", page_icon="🎧", layout="wide")
st.title("🎧 Reference Vibe & Vocal Matcher")
st.caption("Reference song se reverb, delay, kHz brightness aur dynamics ka andaaza lekar vocal par lagata hai.")


def sanitize_audio(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def load_audio_file(uploaded_file, sr: int = 22050) -> np.ndarray:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        y, _ = librosa.load(tmp_path, sr=sr, mono=True)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return sanitize_audio(y)


def analyze_reference_vibe(ref: np.ndarray, sr: int) -> dict:
    ref = sanitize_audio(ref)
    defaults = {
        "reverb_wet": 0.18,
        "reverb_room": 0.42,
        "delay_mix": 0.06,
        "delay_feedback": 0.08,
        "comp_thresh": -18.0,
        "comp_ratio": 2.8,
        "high_shelf_khz": 6.5,
        "high_shelf_db": 1.5,
    }
    if len(ref) < sr:
        return defaults

    rms_frames = librosa.feature.rms(y=ref, frame_length=2048, hop_length=512)[0]
    active = rms_frames[rms_frames > np.max(rms_frames) * 0.15] if len(rms_frames) > 0 else np.array([0.05])
    mean_rms = float(np.mean(active)) if len(active) > 0 else 0.05
    peak = float(np.max(np.abs(ref))) + 1e-7
    crest_db = float(20.0 * np.log10(peak / max(mean_rms, 1e-6)))

    comp_ratio = float(np.clip(2.0 + (18.0 - crest_db) * 0.14, 1.8, 4.2))
    comp_thresh = float(np.clip(-22.0 + (14.0 - crest_db) * 0.4, -26.0, -14.0))

    low_mask = rms_frames < np.percentile(rms_frames, 35)
    low_energy = float(np.mean(rms_frames[low_mask])) if np.any(low_mask) else 0.0
    tail_ratio = float(np.clip(low_energy / max(mean_rms, 1e-6), 0.0, 1.0))
    reverb_wet = float(np.clip(0.10 + tail_ratio * 0.42, 0.08, 0.34))
    reverb_room = float(np.clip(0.30 + tail_ratio * 0.70, 0.25, 0.70))

    onset_env = librosa.onset.onset_strength(y=ref, sr=sr)
    if len(onset_env) > 64:
        autocorr = librosa.autocorrelate(onset_env, max_size=min(len(onset_env) - 1, 100))
        window = autocorr[4:14]
        echo_score = float(np.clip(np.max(window) / (np.max(autocorr) + 1e-6), 0.0, 1.0)) if np.max(autocorr) > 0 else 0.0
    else:
        echo_score = 0.0

    delay_mix = float(np.clip(0.03 + echo_score * 0.09, 0.02, 0.12))
    delay_feedback = float(np.clip(0.05 + echo_score * 0.09, 0.04, 0.14))

    # Spectral centroid -> kHz brightness estimate
    centroid_hz = float(np.mean(librosa.feature.spectral_centroid(y=ref, sr=sr)))
    ref_khz = round(float(np.clip(centroid_hz / 1000.0, 1.5, 10.0)), 2)
    high_shelf_khz = round(float(np.clip(ref_khz * 1.6, 4.0, 9.0)), 1)
    high_shelf_db = round(float(np.clip((ref_khz - 2.5) * 0.8, -2.0, 4.0)), 1)

    return {
        "reverb_wet": round(reverb_wet, 2),
        "reverb_room": round(reverb_room, 2),
        "delay_mix": round(delay_mix, 2),
        "delay_feedback": round(delay_feedback, 2),
        "comp_thresh": round(comp_thresh, 1),
        "comp_ratio": round(comp_ratio, 1),
        "ref_khz": ref_khz,
        "high_shelf_khz": high_shelf_khz,
        "high_shelf_db": high_shelf_db,
    }


def apply_vocal_vibe(vocal: np.ndarray, sr: int, p: dict, hp_freq: float, vocal_gain_db: float) -> np.ndarray:
    vocal = sanitize_audio(vocal)
    if len(vocal) == 0 or np.max(np.abs(vocal)) < 1e-6:
        return vocal

    try:
        import noisereduce as nr
        vocal = sanitize_audio(nr.reduce_noise(y=vocal, sr=sr, prop_decrease=0.45, stationary=True))
    except Exception:
        pass

    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=hp_freq),
        HighShelfFilter(cutoff_frequency_hz=float(p["high_shelf_khz"]) * 1000.0, gain_db=float(p["high_shelf_db"])),
        Compressor(threshold_db=float(p["comp_thresh"]), ratio=float(p["comp_ratio"]), attack_ms=8.0, release_ms=120.0),
        Delay(delay_seconds=0.18, feedback=float(p["delay_feedback"]), mix=float(p["delay_mix"])),
        Reverb(room_size=float(p["reverb_room"]), damping=0.55, wet_level=float(p["reverb_wet"]), dry_level=0.85, width=0.8),
        Gain(gain_db=vocal_gain_db),
        Limiter(threshold_db=-1.0, release_ms=100.0),
    ])
    return sanitize_audio(board(vocal, sr))


if "vibe_params" not in st.session_state:
    st.session_state.vibe_params = analyze_reference_vibe(np.zeros(22050, dtype=np.float32), 22050)

col1, col2 = st.columns(2)
with col1:
    ref_file = st.file_uploader("🎧 Reference Song Upload Karo", type=["wav", "mp3", "ogg", "m4a"])
with col2:
    vocal_file = st.file_uploader("🎤 Apni Vocal Track Upload Karo", type=["wav", "mp3", "ogg", "m4a"])

if st.button("🔍 Reference Analyze Karo", use_container_width=True):
    if ref_file is None:
        st.warning("Pehle Reference Song upload karo.")
    else:
        with st.spinner("Reference song analyze ho raha hai..."):
            ref_audio = load_audio_file(ref_file, sr=22050)
            st.session_state.vibe_params = analyze_reference_vibe(ref_audio, sr=22050)
        st.success("Reference analysis taiyaar hai!")

p = st.session_state.vibe_params
st.info(
    f"Reference Andaaza — Center: ~{p.get('ref_khz', 2.5)} kHz | "
    f"Reverb: {p['reverb_wet']:.2f} | Delay: {p['delay_mix']:.2f} | "
    f"Comp: {p['comp_ratio']:.1f}:1 ({p['comp_thresh']:.1f} dB) | High Shelf: {p['high_shelf_db']:+.1f} dB @ {p['high_shelf_khz']} kHz"
)

hp_freq = st.slider("Highpass Filter (Hz)", 50.0, 200.0, 95.0, 5.0)
vocal_gain_db = st.slider("Vocal Output Level (dB)", -12.0, 12.0, 2.0, 0.5)
out_name = st.text_input("Output Vocal Name", value="Vocal_Ref_Vibe")

if st.button("✨ Reference Vibe Vocal Par Lagao", type="primary", use_container_width=True):
    if vocal_file is None:
        st.warning("Pehle apni Vocal Track upload karo.")
    else:
        with st.spinner("Vocal par reference vibe lagayi ja rahi hai..."):
            sr = 44100
            if ref_file is not None:
                ref_audio = load_audio_file(ref_file, sr=22050)
                p = analyze_reference_vibe(ref_audio, sr=22050)
                st.session_state.vibe_params = p

            vocal_raw = load_audio_file(vocal_file, sr=sr)
            vocal_out = apply_vocal_vibe(vocal_raw, sr=sr, p=p, hp_freq=hp_freq, vocal_gain_db=vocal_gain_db)

            if np.max(np.abs(vocal_out)) < 1e-6:
                st.error("Output audio silent aaya.")
            else:
                buf = io.BytesIO()
                sf.write(buf, vocal_out, sr, format="WAV")
                buf.seek(0)
                safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in out_name.strip()) or "Vocal_Mix"
                st.audio(buf.getvalue(), format="audio/wav")
                st.download_button("⬇️ Download Vibe Vocal (WAV)", buf.getvalue(), f"{safe}.wav", "audio/wav", use_container_width=True)
