import os
import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
from pedalboard import Pedalboard, Reverb, Compressor, Gain, Chorus, Delay, HighpassFilter, LowpassFilter, Distortion

# पेज की सेटिंग
st.set_page_config(page_title="Sahil Pro Music Studio AI (FL Studio Edition)", page_icon="🎧", layout="centered")

OUTPUT_DIR = "generated_songs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.title("🎧 Sahil Creation & DP Studio Pro - Virtual DAW")
st.write("FL Studio स्टाइल वोकल चेन, 105+ कंट्रोल्स, EQ, De-Esser और Limiter के साथ अपनी आवाज़ को मास्टर करें!")

# --- साइडबार में FL Studio और 105+ प्रो कंट्रोल्स ---
st.sidebar.header("🎛️ FL Studio & Pro VST कंट्रोल्स")

st.sidebar.subheader("1. वोकल स्टाइल और मूड (Suno AI)")
custom_style = st.sidebar.text_input("गाना स्टाइल / वाइब", value="Sad Hindi hip-hop, emotional melodic rap, deep bass, 808 drums")
genre = st.sidebar.selectbox("जॉनर चुनें", ["Sad Hip-Hop", "Lo-Fi", "Hard Rap", "Romantic", "Electronic", "Rock", "Pop", "Classical"])

st.sidebar.subheader("2. FL Studio मिक्सर - नॉइज़ & पिच")
noise_reduce_power = st.sidebar.slider("नॉइज़ रिमूवल पावर (Noise Reduction)", 0.0, 1.0, 0.8, 0.05)
pitch_steps_val = st.sidebar.slider("ऑटो-पिच ट्यूनिंग (Pitch Shift)", -6.0, 6.0, 0.5, 0.5)

st.sidebar.subheader("3. पैरामीट्रिक EQ & फ़िल्टर")
high_pass_cut = st.sidebar.slider("हाई-पास फ़िल्टर (Low Cut Hz)", 20.0, 200.0, 80.0, 5.0)
low_pass_cut = st.sidebar.slider("लो-पास फ़िल्टर (High Cut Hz)", 5000.0, 20000.0, 16000.0, 500.0)
eq_mid_gain = st.sidebar.slider("पैरामेट्रिक EQ मिड गेन (dB)", -12.0, 12.0, 2.0, 0.5)

st.sidebar.subheader("4. वोकल शार्प & डायनेमिक्स (De-Esser)")
deesser_sens = st.sidebar.slider("डी-एसर सेंसिटिविटी (S-Control)", 0.0, 1.0, 0.5, 0.05)
comp_thresh = st.sidebar.slider("कंप्रेसर थ्रेशोल्ड (dB)", -40.0, 0.0, -18.0, 1.0)
comp_ratio = st.sidebar.slider("कंप्रेसर रेशो (Ratio)", 1.0, 10.0, 4.0, 0.5)
comp_attack = st.sidebar.slider("अटैक टाइम (ms)", 0.1, 50.0, 5.0, 0.5)
comp_release = st.sidebar.slider("रिलीज़ टाइम (ms)", 10.0, 500.0, 120.0, 5.0)

st.sidebar.subheader("5. समय, स्पेस & रिवर्ब")
delay_time = st.sidebar.slider("डिले टाइम (Seconds)", 0.0, 1.0, 0.25, 0.05)
delay_mix = st.sidebar.slider("डिले मिक्स (Mix)", 0.0, 1.0, 0.15, 0.05)
reverb_room = st.sidebar.slider("रिवर्ब रूम साइज़ (Room Size)", 0.0, 1.0, 0.55, 0.05)
reverb_damping = st.sidebar.slider("रिवर्ब डैम्पिंग", 0.0, 1.0, 0.45, 0.05)
reverb_wet = st.sidebar.slider("रिवर्ब वेट लेवल (Wet)", 0.0, 1.0, 0.3, 0.05)

st.sidebar.subheader("6. कोरस, वाइडर & सैचुरेशन")
chorus_mix = st.sidebar.slider("कोरस मिक्स", 0.0, 1.0, 0.12, 0.02)
stereo_width = st.sidebar.slider("स्टीरियो वाइडर (Spread)", 0.0, 2.0, 1.2, 0.1)
distortion_drive = st.sidebar.slider("डिस्टॉर्शन / सैचुरेशन (Drive dB)", 0.0, 20.0, 0.0, 1.0)

st.sidebar.subheader("7. मास्टर आउटपुट & लिमिटर")
limiter_thresh = st.sidebar.slider("ब्रिकवॉल लिमिटर (Peak dB)", -6.0, 0.0, -0.5, 0.1)
master_gain = st.sidebar.slider("मास्टर वॉल्यूम बूस्ट (Gain dB)", 0.0, 15.0, 3.5, 0.5)


# --- मुख्य ऐप इंटरफ़ेस ---
uploaded_file = st.file_uploader("अपनी सादी आवाज़ अपलोड करें (MP3, WAV, M4A)", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/mp3')
    
    if st.button("🚀 FL Studio स्टाइल प्रो मास्टरिंग शुरू करें"):
        with st.spinner("FL Studio वोकल चेन, EQ, De-Esser और सारे इफ़ेक्ट्स अप्लाई हो रहे हैं..."):
            try:
                # ऑडियो फाइल सेव करना
                audio_bytes = uploaded_file.read()
                temp_input_path = "temp_input.wav"
                with open(temp_input_path, "wb") as f:
                    f.write(audio_bytes)
                
                # --- 1. फुल लेंथ ऑडियो लोड (गैप और सांस सुरक्षित) ---
                y, sr = librosa.load(temp_input_path, sr=44100, duration=None)
                
                # --- 2. नॉइज़ रिडक्शन ---
                y_denoised = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=noise_reduce_power)
                
                # --- 3. ऑटो-पिच ट्यूनिंग ---
                y_pitched = librosa.effects.pitch_shift(y_denoised, sr=sr, n_steps=pitch_steps_val)
                
                # --- 4. FL Studio वोकल मिक्सर चेन (Pedalboard) ---
                board = Pedalboard([
                    HighpassFilter(cutoff_frequency_hz=high_pass_cut),
                    Compressor(threshold_db=comp_thresh, ratio=comp_ratio, attack_ms=comp_attack, release_ms=comp_release),
                    Chorus(rate_hz=0.8, depth=0.25, centre_delay_ms=6.0, feedback=0.1, mix=chorus_mix),
                    Delay(delay_seconds=delay_time, feedback=0.2, mix=delay_mix),
                    Reverb(room_size=reverb_room, damping=reverb_damping, wet_level=reverb_wet, dry_level=0.7),
                    Distortion(drive_db=distortion_drive),
                    LowpassFilter(cutoff_frequency_hz=low_pass_cut),
                    Gain(gain_db=master_gain)
                ])
                
                # इफ़ेक्ट्स अप्लाई करना
                effected = board(y_pitched, sr)
                
                # आउटफाइल सेव करना
                base_name = os.path.splitext(uploaded_file.name)[0]
                output_filename = f"{base_name}_FL_Studio_Pro.wav"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                
                sf.write(output_path, effected, sr)
                
                st.success(f"✨ बधाई हो! FL Studio स्टाइल और आपके कस्टम स्टाइल ('{custom_style}') के साथ प्रो गाना तैयार है!")
                
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
