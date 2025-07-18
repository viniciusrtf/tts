import os
import re
import argparse
import subprocess
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

def parse_transcription(file_path):
    """
    Parses a transcription file and extracts speaker and dialogue.
    Expected format: [0.0s–2.2s] (SPEAKER_00) Text
    """
    lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.search(r'\(SPEAKER_(\d+)\)\s*(.*)', line)
            if match:
                speaker_id = int(match.group(1))
                text = match.group(2).strip()
                lines.append({'speaker_id': speaker_id, 'text': text})
    return lines

def run_ffmpeg(input_path, output_path, speed_factor):
    """
    Applies a speed factor to an audio file using ffmpeg.
    """
    if speed_factor == 1.0:
        os.rename(input_path, output_path)
        return

    command = [
        'ffmpeg',
        '-i', input_path,
        '-filter:a', f"atempo={speed_factor}",
        '-y',
        output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print("Error running ffmpeg:")
        print(e.stderr.decode())
        raise
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

def main():
    """
    Main function to generate speech from a transcription file.
    """
    parser = argparse.ArgumentParser(description="Generate speech from a transcription file using different speakers with ChatterboxTTS.")
    parser.add_argument('-t', '--transcription', type=str, required=True, help="Path to the transcription file.")
    parser.add_argument('-r', '--references', nargs='+', required=True, help="List of reference audio files for speakers (e.g., speaker0.wav speaker1.wav).")
    parser.add_argument('-s', '--speed', type=float, default=1.0, help="Speed factor for the output audio (e.g., 1.2 for 20% faster).")
    parser.add_argument('--exaggeration', type=float, default=0.6, help="Exaggeration factor for TTS generation.")
    parser.add_argument('--cfg_weight', type=float, default=0.7, help="CFG weight for TTS generation.")


    args = parser.parse_args()

    print("Loading ChatterboxTTS model on CPU...")
    model = ChatterboxTTS.from_pretrained(device="cpu")

    dialogue_lines = parse_transcription(args.transcription)
    if not dialogue_lines:
        print("No dialogue lines found in the transcription file.")
        return

    output_dir = os.path.dirname(os.path.abspath(args.transcription))

    for i, line in enumerate(dialogue_lines):
        speaker_id = line['speaker_id']
        text = line['text']

        if speaker_id >= len(args.references):
            print(f"Warning: Speaker ID {speaker_id} is out of range for the provided reference files. Skipping line {i+1}.")
            continue

        reference_wav = args.references[speaker_id]
        output_filename_temp = os.path.join(output_dir, f"{i:03d}_temp.wav")
        output_filename_final = os.path.join(output_dir, f"{i:03d}.wav")

        print(f"Generating line {i+1}/{len(dialogue_lines)}: Speaker {speaker_id} -> '{text[:50]}...' ")

        wav = model.generate(
            text,
            audio_prompt_path=reference_wav,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight
        )

        ta.save(output_filename_temp, wav, model.sr)

        print(f"Applying speed factor of {args.speed}...")
        run_ffmpeg(output_filename_temp, output_filename_final, args.speed)

    print("\nDone. All audio files have been generated.")

if __name__ == '__main__':
    main()