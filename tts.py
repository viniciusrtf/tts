#!/usr/bin/env python3
"""
tts_from_segments.py
--------------------
Generate per‑segment speech files from a JSON transcription produced by
WhisperX/any diarization tool in the form:

{
  "segments": [
    {
      "id": 0,
      "start": 2.866,
      "end": 15.285,
      "text": " … ",
      "speaker": "SPEAKER_00"
    },
    …
  ],
  "language": "en"
}

For every segment we:

* pick the correct reference voice (index = numeric part of "SPEAKER_XX")
* synthesise speech with ChatterboxTTS
* write `<idx>.wav` next to the JSON file
* write a companion `_manifest.txt` with timing info ↔ audio‑path mapping
"""

import os
import re
import json                           #  ← NEW
import argparse
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS


# ────────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ────────────────────────────────────────────────────────────────────────────────
def speaker_to_id(speaker_str: str) -> int:
    """
    Convert "SPEAKER_00" → 0 (int). Returns -1 if no digits found.
    """
    m = re.search(r'(\d+)$', speaker_str)
    return int(m.group(1)) if m else -1


def parse_transcription_json(file_path):
    """
    Read the JSON file and return a list of dicts compatible with the original
    code: start_time, end_time, speaker_id, text
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    lines = []

    for seg in segments:
        start_time = float(seg["start"])
        end_time   = float(seg["end"])
        text       = seg.get("text", "").strip()

        speaker_raw = seg.get("speaker", "SPEAKER_00")
        speaker_id  = speaker_to_id(speaker_raw)

        lines.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "speaker_id": speaker_id,
                "text": text,
            }
        )

    return lines


# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate speech from a JSON transcription file using ChatterboxTTS."
    )
    parser.add_argument(
        "-t",
        "--transcription",
        type=str,
        required=True,
        help="Path to the JSON transcription file.",
    )
    parser.add_argument(
        "-r",
        "--references",
        nargs="+",
        required=True,
        help="Reference audio files for speakers (order must match SPEAKER_00, SPEAKER_01 …).",
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.6,
        help="Exaggeration factor for TTS generation.",
    )
    parser.add_argument(
        "--cfg_weight",
        type=float,
        default=0.7,
        help="CFG weight for TTS generation.",
    )

    args = parser.parse_args()

    print("Loading ChatterboxTTS model on CPU …")
    model = ChatterboxTTS.from_pretrained(device="cpu")

    dialogue_lines = parse_transcription_json(args.transcription)   # ← CHANGED
    if not dialogue_lines:
        print("No segments found in the transcription file.")
        return

    output_dir = os.path.dirname(os.path.abspath(args.transcription))
    manifest_data = []

    for i, line in enumerate(dialogue_lines):
        speaker_id = line["speaker_id"]
        text = line["text"]

        if speaker_id < 0 or speaker_id >= len(args.references):
            print(
                f"Warning: Speaker ID {speaker_id} has no matching reference audio. "
                f"Skipping segment {i + 1}."
            )
            continue

        reference_wav = args.references[speaker_id]
        out_wav = os.path.join(output_dir, f"{i:03d}.wav")

        preview = (text[:50] + "…") if len(text) > 50 else text
        print(f"[{i + 1:3}/{len(dialogue_lines)}] SPK {speaker_id:02d} → “{preview}”")

        wav = model.generate(
            text,
            audio_prompt_path=reference_wav,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
        )

        ta.save(out_wav, wav, model.sr)

        manifest_data.append(
            {
                "start_time": line["start_time"],
                "end_time": line["end_time"],
                "speaker_id": speaker_id,
                "audio_path": os.path.abspath(out_wav),
            }
        )

    # Manifest
    base = os.path.splitext(args.transcription)[0]
    manifest_path = f"{base}_manifest.txt"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for item in manifest_data:
            f.write(
                f"[{item['start_time']:.3f}s–{item['end_time']:.3f}s] "
                f"(SPEAKER_{item['speaker_id']:02d}) {item['audio_path']}\n"
            )

    print(f"\nManifest written to: {manifest_path}")
    print("Done – all audio files generated.")


if __name__ == "__main__":
    main()

