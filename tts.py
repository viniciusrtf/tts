import torchaudio as ta
from chatterbox.tts import ChatterboxTTS   # ← official package

# 1) load on CPU
model = ChatterboxTTS.from_pretrained(device="cpu")   # no CUDA used

# 2) generate from a 6–10 s reference clip
wav = model.generate(
    "Because, TODAY, that guy pays rent... but what about the day he has a house to rent to others?",
    audio_prompt_path="/home/vinicius/Videos/MLnAtcH3kis/reference.wav",   # clean mono WAV, 24 kHz preferred
    exaggeration=0.6,
    cfg_weight=0.7
)

# 3) save
ta.save("/home/vinicius/Videos/MLnAtcH3kis/06.wav", wav, model.sr)
print("✓ wav written")

