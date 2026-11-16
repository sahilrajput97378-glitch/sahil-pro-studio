import os
from flask import Flask, render_template_string, request, send_file
import librosa
import soundfile as sf
import numpy as np
from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter, Gain

app = Flask(__name__)

HTML_TEMPLATE = '''
<!doctype html>
<html lang="hi">
<head>
    <meta charset="utf-8">
    <title>Smart AI Vocal Match Studio</title>
    <style>
        body { font-family: Arial, sans-serif; background: #121212; color: #fff; padding: 20px; display: flex; justify-content: center; }
        .container { width: 100%; max-width: 500px; background: #1e1e1e; padding: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
        h2 { text-align: center; color: #1db954; margin-bottom: 20px; }
        label { display: block; margin-top: 15px; font-weight: bold; font-size: 14px; }
        input[type="file"], textarea { width: 100%; margin-top: 5px; padding: 10px; border-radius: 6px; border: 1px solid #444; background: #2a2a2a; color: #fff; box-sizing: border-box; }
        button { width: 100%; margin-top: 25px; padding: 12px; border-radius: 6px; background: #1db954; color: #fff; font-weight: bold; font-size: 16px; border: none; cursor: pointer; }
        button:hover { background: #1aa34a; }
    </style>
</head>
<body>
    <div class="container">
        <h2>🎛️ Smart AI Reference Vocal Studio</h2>
        <form method="POST" enctype="multipart/form-data">
            <label>1. ओरिजिनल बीट (Beat):</label>
            <input type="file" name="beat" accept=".wav,.mp3" required>

            <label>2. कच्ची आवाज़ (Dry Vocals):</label>
            <input type="file" name="vocal" accept=".wav,.mp3" required>

            <label>3. रेफरेंस सॉन्ग (वाइब और इफ़ेक्ट्स मैच करने के लिए):</label>
            <input type="file" name="reference" accept=".wav,.mp3">

            <label>4. कस्टम स्टाइल बॉक्स (अपना मूड या वाइब लिखें):</label>
            <textarea name="style" rows="3" placeholder="जैसे: Sad Haryanvi vibe, deep reverb, clean vocals..."></textarea>

            <button type="submit">🚀 स्मार्ट रेफरेंस मिक्सिंग शुरू करें</button>
        </form>
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        beat_file = request.files['beat']
        vocal_file = request.files['vocal']
        ref_file = request.files.get('reference')
        style_text = request.form.get('style', '').lower()

        os.makedirs('uploads', exist_ok=True)
        beat_path = os.path.join('uploads', beat_file.filename)
        vocal_path = os.path.join('uploads', vocal_file.filename)
        beat_file.save(beat_path)
        vocal_file.save(vocal_path)

        # ऑडियो लोड करना
        vocal_audio, sr = librosa.load(vocal_path, sr=22050)
        beat_audio, _ = librosa.load(beat_path, sr=22050)

        # डिफॉल्ट इफ़ेक्ट पैरामीटर्स
        rev_room = 0.3
        rev_wet = 0.25
        comp_ratio = 3.0

        # --- स्मार्ट रेफरेंस एनालिसिस लॉजिक ---
        if ref_file and ref_file.filename != '':
            ref_path = os.path.join('uploads', ref_file.filename)
            ref_file.save(ref_path)
            ref_audio, _ = librosa.load(ref_path, sr=22050)
            
            # रेफरेंस गाने की एनर्जी और स्पेक्ट्रल ब्राइटनेस कैलकुलेट करना
            ref_rms = np.mean(librosa.feature.rms(y=ref_audio))
            
            # वाइब के हिसाब से ऑटोमैटिक पैरामीटर सेट करना
            if ref_rms > 0.1:  # अगर गाना एनर्जेटिक/लाउड है
                comp_ratio = 4.5
                rev_wet = 0.20
            else:  # अगर स्लो/सैड/मेलोडी वाइब है (ज्यादा रिवर्ब चाहिए)
                rev_room = 0.6
                rev_wet = 0.45
                comp_ratio = 2.5
        
        # --- कस्टम स्टाइल बॉक्स का असर ---
        if 'sad' in style_text or 'reverb' in style_text:
            rev_room = 0.65
            rev_wet = 0.5
        if 'clean' in style_text or 'dry' in style_text:
            rev_wet = 0.15

        # स्मार्ट पैडलबोर्ड जो रेफरेंस और स्टाइल के हिसाब से एडजस्ट होगा
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=85.0),
            Compressor(threshold_db=-20.0, ratio=comp_ratio),
            Reverb(room_size=rev_room, damping=0.5, wet_level=rev_wet, dry_level=1.0 - rev_wet),
            Gain(gain_db=2.0)
        ])

        # वोकल्स पर ऑटोमैटिक प्रोसेसिंग
        processed_vocal = board(vocal_audio, sr)

        # बीट और वोकल का मिक्सअप
        min_len = min(len(processed_vocal), len(beat_audio))
        final_mix = processed_vocal[:min_len] + 0.85 * beat_audio[:min_len]

        output_path = os.path.join('uploads', 'final_mastered_song.wav')
        sf.write(output_path, final_mix, sr)

        return send_file(output_path, as_attachment=True)

    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
