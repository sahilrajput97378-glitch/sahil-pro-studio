import os
import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
from pedalboard import (
    Pedalboard,
    Reverb,
    Compressor,
    Gain,
    Delay,
    HighpassFilter,
    Limiter,
)

st.set_page_config(
    page_title="Sahil Pro AI Smart Studio DAW",
    page_icon="🎧",
    layout="centered",
)

OUTPUT_DIR = "generated_songs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.title("🎧 Sahil Creation & DP Studio Pro - AI Smart Reference Matching")
st.write(
    "रेफरेंस ऑडियो को एनालाइज़ करके रिवर्ब, डिले, कंप्रेशन और गेन का अंदाज़ा लगाकर "
    "आपकी आवाज़ पर स्टूडियो इफेक्ट्स अप्लाई करें।"
)

# --- साइडबार ---
st.sidebar.header("🎛️ AI Smart Reference Settings")
custom_style = st.sidebar.text_input(
    "गाना स्टाइल / वाइब",
    value="Sad Hindi Hip-Hop Melodic Rap",
)
reference_audio_file = st.sidebar.file_uploader(
    "अपना रेफरेंस ट्रैक / वोकल अपलोड करें",
    type=["mp3", "wav", "m4a", "ogg"],
)

auto_match = st.sidebar.checkbox("🤖 AI ऑटो-मैच इफेक्ट्स ऑन रखें", value=True)
if not auto_match:
    manual_reverb_pct = st.sidebar.slider("रिवर्ब (%)", 0, 80, 35)
    manual_delay_pct = st.sidebar.slider("डिले (%)", 0, 50, 20)
    manual_gain_db = st.sidebar.slider("एक्स्ट्रा गेन (dB)", -6.0, 18.0, 4.0, step=0.5)

# --- मुख्य ऐप इंटरफ़ेस ---
uploaded_file = st.file_uploader(
    "अपनी सादी आवाज़ अपलोड करें (MP3, WAV, M4A, OGG)",
    type=["mp3", "wav", "m4a", "ogg"],
)


def safe_rms(y: np.ndarray) -> float:
    if y is None or y.size == 0:
        return 0.0
    val = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
    return val if np.isfinite(val) else 0.0


def analyze_reference_effects(y_ref: np.ndarray, sr: int) -> dict:
    """
    रेफरेंस ऑडियो से RMS लाउडनेस, ट्रांजिएंट/एक्टिविटी और
    टेल-डिके (reverb/echo sustain) का अनुमान निकालता है।
    """
    ref_rms = safe_rms(y_ref)
    if ref_rms <= 1e-7 or y_ref.size < sr // 2:
        return {
            "ref_rms": ref_rms,
            "reverb_mix": 0.35,
            "delay_mix": 0.20,
            "room_size": 0.60,
            "delay_sec": 0.28,
            "comp_ratio": 3.5,
        }

    # फ्रेम-वाइज़ एनर्जी निकालना
    frame_len = max(256, int(0.05 * sr))
    hop_len = max(128, int(0.025 * sr))
    rms_frames = librosa.feature.rms(
        y=y_ref, frame_length=frame_len, hop_length=hop_len
    )[0]
    rms_frames = np.nan_to_num(rms_frames, nan=0.0, posinf=0.0, neginf=0.0)

    peak_rms = float(np.max(rms_frames)) if rms_frames.size > 0 else ref_rms
    mean_rms = float(np.mean(rms_frames)) if rms_frames.size > 0 else ref_rms

    # अगर आवाज़/सिग्नल में सूखने (decay) के बाद भी एनर्जी बनी रहती है,
    # तो रिवर्ब/टेल ज़्यादा होने का संकेत मिलता है
    sustain_ratio = float(np.clip(mean_rms / max(peak_rms, 1e-7), 0.0, 1.0))

    # कम एनर्जी वाले हिस्सों की तुलना में टेल-सस्टेन का अनुमान
    quiet_threshold = np.percentile(rms_frames, 35) if rms_frames.size > 0 else 0.0
    active_threshold = np.percentile(rms_frames, 75) if rms_frames.size > 0 else 0.0
    tail_ratio = float(
        np.clip(quiet_threshold / max(active_threshold, 1e-7), 0.0, 1.0)
    )

    # स्पेक्ट्रल फ्लक्स / ट्रांजिएंट एक्टिविटी (डिले और डायनामिक्स के लिए)
    onset_env = librosa.onset.onset_strength(y=y_ref, sr=sr, hop_length=hop_len)
    onset_env = np.nan_to_num(onset_env, nan=0.0, posinf=0.0, neginf=0.0)
    onset_norm = float(
        np.clip(np.mean(onset_env) / max(np.max(onset_env), 1e-6), 0.0, 1.0)
    )

    reverb_score = 0.55 * sustain_ratio + 0.45 * tail_ratio
    reverb_mix = float(np.clip(0.18 + reverb_score * 0.45, 0.18, 0.60))
    room_size = float(np.clip(0.35 + reverb_score * 0.55, 0.35, 0.88))

    # स्पष्ट ट्रांजिएंट/बीट्स होने पर डिले हल्का और कंट्रोल्ड रखते हैं
    delay_mix = float(np.clip(0.12 + (1.0 - onset_norm) * 0.22, 0.12, 0.32))
    delay_sec = float(np.clip(0.22 + (1.0 - sustain_ratio) * 0.16, 0.20, 0.38))

    # डायनामिक रेंज के हिसाब से कंप्रेशन रेशियो
    max_abs = float(np.max(np.abs(y_ref)))
    crest_factor = max_abs / max(ref_rms, 1e-7)
    comp_ratio = float(np.clip(2.5 + (crest_factor - 4.0) * 0.35, 2.5, 5.5))

    return {
        "ref_rms": ref_rms,
        "reverb_mix": reverb_mix,
        "delay_mix": delay_mix,
        "room_size": room_size,
        "delay_sec": delay_sec,
        "comp_ratio": comp_ratio,
    }


if uploaded_file is not None:
    st.audio(uploaded_file)

    if st.button("🚀 AI स्मार्ट मास्टरिग शुरू करें"):
        if reference_audio_file is None:
            st.warning("कृपया साइडबार में एक **रेफरेंस ट्रैक** जरूर अपलोड करें।")
        else:
            with st.spinner("ऑडियो एनालाइज़ और प्रोसेस हो रहा है, कृपया इंतज़ार करें..."):
                user_path = os.path.join(OUTPUT_DIR, "_temp_user_input.wav")
                ref_path = os.path.join(OUTPUT_DIR, "_temp_ref_input.wav")

                try:
                    with open(user_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    with open(ref_path, "wb") as f:
                        f.write(reference_audio_file.getbuffer())

                    sr = 44100
                    y_user, _ = librosa.load(user_path, sr=sr, mono=True)
                    y_ref, _ = librosa.load(ref_path, sr=sr, mono=True)

                    if y_user is None or y_user.size == 0:
                        st.error("अपलोड की गई आवाज़ वाली फाइल खाली या खराब लग रही है।")
                        st.stop()

                    if y_ref is None or y_ref.size == 0:
                        st.error("रेफरेंस फाइल खाली या खराब लग रही है।")
                        st.stop()

                    # 1. हल्की नॉइज़ क्लीनअप (वोकल को ज़्यादा पतला किए बिना)
                    try:
                        y_clean = nr.reduce_noise(
                            y=y_user,
                            sr=sr,
                            stationary=True,
                            prop_decrease=0.55,
                        )
                    except Exception:
                        y_clean = y_user

                    y_clean = np.nan_to_num(
                        y_clean, nan=0.0, posinf=0.0, neginf=0.0
                    ).astype(np.float32)

                    # 2. रेफरेंस एनालिसिस और गेन मैचिंग
                    ref_stats = analyze_reference_effects(y_ref, sr)
                    user_rms = safe_rms(y_clean)
                    ref_rms = ref_stats["ref_rms"]

                    if user_rms > 1e-6 and ref_rms > 1e-6:
                        raw_db = 20.0 * np.log10(ref_rms / user_rms)
                    else:
                        raw_db = 4.0

                    if auto_match:
                        reverb_mix = ref_stats["reverb_mix"]
                        delay_mix = ref_stats["delay_mix"]
                        gain_db = float(np.clip(raw_db + 2.0, -4.0, 18.0))
                    else:
                        reverb_mix = float(manual_reverb_pct) / 100.0
                        delay_mix = float(manual_delay_pct) / 100.0
                        gain_db = float(np.clip(raw_db + manual_gain_db, -6.0, 20.0))

                    st.info(
                        f"🤖 AI एनालिसिस: रिवर्ब ≈ {int(round(reverb_mix * 100))}%, "
                        f"डिले ≈ {int(round(delay_mix * 100))}%, "
                        f"गेन = {round(gain_db, 2)} dB, "
                        f"कंप्रेशन = {round(ref_stats['comp_ratio'], 1)}:1"
                    )

                    # 3. स्टूडियो प्रोसेसिंग चेन
                    board = Pedalboard([
                        HighpassFilter(cutoff_frequency_hz=85.0),
                        Compressor(
                            threshold_db=-20.0,
                            ratio=ref_stats["comp_ratio"],
                            attack_ms=6.0,
                            release_ms=110.0,
                        ),
                        Delay(
                            delay_seconds=ref_stats["delay_sec"],
                            feedback=0.20,
                            mix=delay_mix,
                        ),
                        Reverb(
                            room_size=ref_stats["room_size"],
                            damping=0.45,
                            wet_level=reverb_mix,
                            dry_level=0.78,
                        ),
                        Gain(gain_db=gain_db),
                        Limiter(threshold_db=-0.5),
                    ])

                    effected = board(y_clean, sr)
                    effected = np.nan_to_num(
                        effected, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    effected = np.clip(effected, -1.0, 1.0).astype(np.float32)

                    # 4. सुरक्षित फाइल सेव करना
                    safe_base = "".join(
                        c if c.isalnum() or c in (" ", "-", "_") else "_"
                        for c in os.path.splitext(uploaded_file.name)[0]
                    ).strip() or "vocal_track"

                    output_filename = f"{safe_base}_Studio_Mix.wav"
                    output_path = os.path.join(OUTPUT_DIR, output_filename)

                    sf.write(output_path, effected, sr, subtype="PCM_16")
                    st.success("✨ प्रोसेसिंग पूरी हो गई! नीचे लाइब्रेरी में सुनकर देख।")

                except Exception as e:
                    st.error(f"प्रोसेसिंग में दिक्कत आई: {e}")

                finally:
                    for temp_file in (user_path, ref_path):
                        if os.path.exists(temp_file):
                            try:
                                os.remove(temp_file)
                            except OSError:
                                pass

# --- सॉन्ग लाइब्रेरी और डाउनलोड सेक्शन ---
st.markdown("---")
st.subheader("📂 आपकी सॉन्ग लाइब्रेरी (Generated Songs)")

if os.path.exists(OUTPUT_DIR):
    saved_songs = [
        f
        for f in os.listdir(OUTPUT_DIR)
        if not f.startswith("_temp_") and f.endswith((".wav", ".mp3"))
    ]
else:
    saved_songs = []

if saved_songs:
    for song in sorted(saved_songs, reverse=True):
        song_path = os.path.join(OUTPUT_DIR, song)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text(song)
            st.audio(song_path)
        with col2:
            with open(song_path, "rb") as file:
                st.download_button(
                    label="⬇️ डाउनलोड",
                    data=file.read(),
                    file_name=song,
                    mime="audio/wav",
                    key=f"dl_{song}",
                )
else:
    st.info("अभी तक कोई गाना नहीं बनाया गया है।")
