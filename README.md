# tts.py

This script uses the ChatterboxTTS model to generate speech from a transcription file. It supports multiple speakers and generates a manifest file that maps the generated audio to the original timestamps.

## Usage

### Prerequisites

Before running the script, you need to have the required libraries installed. You can install them using pip:

```bash
pip install chatterbox-tts torchaudio
```

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
*   `--exaggeration`: (Optional) Exaggeration factor for TTS generation. Defaults to `0.6`.
*   `--cfg_weight`: (Optional) CFG weight for TTS generation. Defaults to `0.7`.

### Output

The script will generate audio files (e.g., `000.wav`, `001.wav`, etc.) in the same directory as the transcription file. It will also create a manifest file named `<transcription_file_base>_manifest.txt` with the following format:

```
[0.0s–2.2s] (SPEAKER_00) /path/to/000.wav
[2.5s–5.0s] (SPEAKER_01) /path/to/001.wav
```

### Example

```bash
python tts.py -t dialogue.txt -r speaker0.wav speaker1.wav
```

This command will generate audio files and a `dialogue_manifest.txt` file in the same directory as `dialogue.txt`.
