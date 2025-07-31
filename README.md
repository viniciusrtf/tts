# tts.py

This script generates per-segment speech files from a JSON transcription file, such as one produced by WhisperX or other diarization tools.

It processes a JSON file containing speech segments, synthesizes audio for each segment using the ChatterboxTTS model, and assigns the correct voice based on speaker labels.

## Features

- Parses JSON transcription files with speaker, text, and timing information.
- Generates individual `.wav` files for each speech segment.
- Creates a `_manifest.txt` file that maps the generated audio files to their original timestamps and speaker IDs, suitable for use in audio editing or further processing.

## Prerequisites

Install the required Python libraries:

```bash
pip install chatterbox-tts torchaudio
```

## Usage

### 1. Prepare your Transcription File

The script requires a JSON file containing a list of speech segments. Each segment must be an object with `start`, `end`, `text`, and `speaker` keys.

**Example `dialogue.json`:**
```json
{
  "segments": [
    {
      "start": 2.86,
      "end": 15.28,
      "text": "This is the first line of dialogue spoken by speaker zero.",
      "speaker": "SPEAKER_00"
    },
    {
      "start": 16.1,
      "end": 22.5,
      "text": "And this is the second line, spoken by a different person.",
      "speaker": "SPEAKER_01"
    }
  ]
}
```

### 2. Prepare Reference Audio

You need at least one high-quality `.wav` file to serve as a voice reference for each speaker (e.g., `speaker0.wav` for `SPEAKER_00`).

### 3. Run the Script

Execute the script from your terminal, providing the path to your JSON transcription and the reference audio files.

```bash
python tts.py -t dialogue.json -r speaker0.wav speaker1.wav
```

## Command-Line Arguments

-   `-t`, `--transcription`: **(Required)** Path to the input JSON transcription file.
-   `-r`, `--references`: **(Required)** A list of reference `.wav` files. The order must correspond to the speaker IDs (e.g., the first file for `SPEAKER_00`, the second for `SPEAKER_01`, and so on).
-   `--exaggeration`: (Optional) The exaggeration factor for TTS generation. Defaults to `0.6`.
-   `--cfg_weight`: (Optional) The CFG weight for TTS generation. Defaults to `0.7`.

## Output

The script generates two sets of outputs in the same directory as your transcription file:

1.  **Numbered Audio Files**: A `.wav` file for each segment (e.g., `000.wav`, `001.wav`, ...).
2.  **Manifest File**: A text file named `<your_transcription>_manifest.txt`.

**Example `dialogue_manifest.txt`:**
```
[2.866s–15.285s] (SPEAKER_00) /path/to/your/project/000.wav
[16.100s–22.500s] (SPEAKER_01) /path/to/your/project/001.wav
```