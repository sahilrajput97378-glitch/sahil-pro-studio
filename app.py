import os
import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
from pedalboard import (
    Pedalboard, Reverb, Compressor, Gain, Chorus, Delay,
    HighpassFilter, LowpassFilter, Distortion, Phaser, Limiter
)

st.set_page_config(page_title="Sahil Pro AI Smart Studio DAW", page_icon="🎧", layout="centered")

OUTPUT_DIR = "generated_songs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.title("🎧 Sahil Creation & DP Studio Pro - AI Smart Reference Matching")
st.write("AI आपके रेफरेंस गाने को एनालाइज करके, आपकी आवाज़ पर बिल्कुल वैसे ही इफ़ेक्ट्स अप्लाई करेगा!")

# --- साइडबार ---
st.sidebar.header("🎛️ AI Smart Reference Settings")
custom_style = st.sidebar.text_input("गाना स्टाइल / वाइब", value="Sad Hindi Hip-Hop Melodic Rap")
reference_audio_file = st.sidebar.file_uploader("अपना रेफरेंस गाना अपलोड करें", type=["mp3", "wav", "m4a"])

# --- मुख्य ऐप इंटरफ़ेस ---
uploaded_file = st.file_uploader("अपनी सादी आवाज़ अपलोड करें (MP3, WAV, M4A)", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/mp3')

    if st.button("🚀 AI स्मार्ट मास्टरिग शुरू करें"):
        if reference_audio_file is None:
            st.warning("कृपया AI मैचिंग के लिए साइडबार में एक **रेफरेंस गाना** जरूर अपलोड करें!")
        else:
            with st.spinner("ऑडियो प्रोसेस हो रहा है, कृपया इंतज़ार करें..."):
                try:
                    # 1. फाइलें सेव करना
                    user_bytes = uploaded_file.read()
                    ref_bytes = reference_audio_file.read()

                    user_path = "temp_user.wav"
                    ref_path = "temp_ref.wav"

                    with open(user_path, "wb") as f:
                        f.write(user_bytes)
                    with open(ref_path, "wb") as f:
                        f.write(ref_bytes)

                    # 2. ऑडियो लोड करना
                    y_user, sr = librosa.load(user_path, sr=44100, mono=True)
                    y_ref, _ = librosa.load(ref_path, sr=44100, mono=True)

                    # 3. RMS वॉल्यूम मैचिंग
                    ref_rms = np.sqrt(np.mean(y_ref**2))
                    user_rms = np.sqrt(np.mean(y_user**2))
                    volume_adjustment_db = 20 * np.log10(ref_rms / user_rms) if user_rms > 0 else 0.0

                    # 4. AI कैलकुलेशन (वही वाली ताकत और स्पेस)
                    ref_activity = np.mean(np.abs(np.diff(y_ref)))
                    ai_reverb_mix = min(0.35, max(0.10, ref_activity * 0.4))
                    ai_delay_mix = min(0.20, max(0.05, ref_activity * 0.25))
                    ai_compression_ratio = 3.5 if ref_activity > 0.02 else 2.5

                    st.info(f"🤖 AI सेटिंग्स: रिवर्ब = {int(ai_reverb_mix*100)}%, डिले = {int(ai_delay_mix*100)}%, गेन = {round(volume_adjustment_db, 2)} dB")

                    # 5. क्लीनिंग
                    y_denoised = nr.reduce_noise(y=y_user, sr=sr, stationary=True, prop_decrease=0.8)

                    # 6. पैडल बोर्ड (वही वाली फुल चेन)
                    ai_matched_board = Pedalboard([
                        HighpassFilter(cutoff_frequency_hz=85.0),
                        Compressor(threshold_db=-18.0, ratio=ai_compression_ratio),
                        Delay(delay_seconds=0.25, feedback=0.15, mix=ai_delay_mix),
                        Reverb(room_size=0.45, damping=0.5, wet_level=ai_reverb_mix),
                        Limiter(threshold_db=-0.8),
                        Gain(gain_db=volume_adjustment_db + 1.5)
                    ])

                    # इफ़ेक्ट्स प्रोसेस करना
                    effected = ai_matched_board(y_denoised, sr)

                    # 7. आउटपुट सेव करना
                    base_name = os.path.splitext(uploaded_file.name)[0]
                    output_filename = f"{base_name}_Full_Matched.wav"
                    output_path = os.path.join(OUTPUT_DIR, output_filename)

                    sf.write(output_path, effected, sr)

                    st.success("✨ आवाज़ बिल्कुल उसी अंदाज़ और ताकत के साथ तैयार हो गई है!")

                except Exception as e:
                    st.error(f"प्रोसेसिंग में एरर आया: {e}")

# सॉन्ग लाइब्रेरी और डाउनलोड सेक्शन
st.markdown("---")
st.subheader("📂 आपकी सॉन्ग लाइब्रेरी (Generated Songs)")

saved_songs = os.listdir(OUTPUT_DIR)
if saved_songs:
    for song in sorted(saved_songs, reverse=True):
        if song.endswith(('.wav', '.mp3')):
            song_path = os.path.join(OUTPUT_DIR, song)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(song)
                st.audio(song_path)
            with col2:
                with open(song_path, "rb") as file:
                    st.download_button(
                        label="⬇️ डाउनलोड",
                        data=file,
                        file_name=song,
                        mime="audio/wav",
                        key=song
                    )
else:
    st.info("अभी तक कोई गाना नहीं बनाया गया है।")
