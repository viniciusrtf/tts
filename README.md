# tts.py

This script uses the ChatterboxTTS model to generate speech from a transcription file. It supports multiple speakers and can adjust the speed of the generated audio.

## Usage

### Prerequisites

Before running the script, you need to have the required libraries installed. You can install them using pip:

```bash
pip install chatterbox-tts torchaudio
```

You also need to have `ffmpeg` installed on your system.

### Running the script

To run the script, you need a transcription file and at least one reference audio file for the speaker(s).

The transcription file should have the following format:

```
[0.0s–2.2s] (SPEAKER_00) Text for speaker 0.
[2.5s–5.0s] (SPEAKER_01) Text for speaker 1.
```

You can then run the script with the following command:

```bash
python tts.py -t <transcription_file> -r <reference_audio_1> <reference_audio_2> ...
```

### Arguments

*   `-t`, `--transcription`: Path to the transcription file.
*   `-r`, `--references`: List of reference audio files for speakers (e.g., `speaker0.wav` `speaker1.wav`). The order of the files corresponds to the speaker ID in the transcription file (SPEAKER_00, SPEAKER_01, etc.).
*   `-s`, `--speed`: (Optional) Speed factor for the output audio (e.g., `1.2` for 20% faster). Defaults to `1.0`.
*   `--exaggeration`: (Optional) Exaggeration factor for TTS generation. Defaults to `0.6`.
*   `--cfg_weight`: (Optional) CFG weight for TTS generation. Defaults to `0.7`.

### Example

```bash
python tts.py -t dialogue.txt -r speaker0.wav speaker1.wav -s 1.1
```

This command will generate audio files in the same directory as the transcription file, one for each line in the file. The audio will be 10% faster than the original.
