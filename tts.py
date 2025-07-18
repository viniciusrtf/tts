import os
import re
import argparse
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

def parse_transcription(file_path):
    """
    Parses a transcription file and extracts timestamps, speaker, and dialogue.
    Expected format: [0.0s–2.2s] (SPEAKER_00) Text
    """
    lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            time_match = re.search(r'\[(\d+\.\d+)s–(\d+\.\d+)s\]', line)
            if not time_match:
                continue

            speaker_match = re.search(r'\(SPEAKER_(\d+)\)\s*(.*)', line)
            if speaker_match:
                start_time = float(time_match.group(1))
                end_time = float(time_match.group(2))
                speaker_id = int(speaker_match.group(1))
                text = speaker_match.group(2).strip()
                lines.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'speaker_id': speaker_id,
                    'text': text
                })
    return lines

def main():
    """
    Main function to generate speech from a transcription file.
    """
    parser = argparse.ArgumentParser(description="Generate speech from a transcription file using different speakers with ChatterboxTTS.")
    parser.add_argument('-t', '--transcription', type=str, required=True, help="Path to the transcription file.")
    parser.add_argument('-r', '--references', nargs='+', required=True, help="List of reference audio files for speakers (e.g., speaker0.wav speaker1.wav).")
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
    manifest_data = []

    for i, line in enumerate(dialogue_lines):
        speaker_id = line['speaker_id']
        text = line['text']

        if speaker_id >= len(args.references):
            print(f"Warning: Speaker ID {speaker_id} is out of range for the provided reference files. Skipping line {i+1}.")
            continue

        reference_wav = args.references[speaker_id]
        output_filename_final = os.path.join(output_dir, f"{i:03d}.wav")

        print(f"Generating line {i+1}/{len(dialogue_lines)}: Speaker {speaker_id} -> '{text[:50]}...' ")

        wav = model.generate(
            text,
            audio_prompt_path=reference_wav,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight
        )

        ta.save(output_filename_final, wav, model.sr)
        
        # Store data for manifest
        manifest_data.append({
            'start_time': line['start_time'],
            'end_time': line['end_time'],
            'speaker_id': line['speaker_id'],
            'audio_path': os.path.abspath(output_filename_final)
        })

    # Generate manifest file
    transcription_path_no_ext = os.path.splitext(args.transcription)[0]
    manifest_path = f"{transcription_path_no_ext}_manifest.txt"
    
    with open(manifest_path, 'w', encoding='utf-8') as f:
        for item in manifest_data:
            f.write(f"[{item['start_time']:.1f}s–{item['end_time']:.1f}s] (SPEAKER_{item['speaker_id']:02d}) {item['audio_path']}\n")

    print(f"\nManifest file generated at: {manifest_path}")
    print("Done. All audio files have been generated.")

if __name__ == '__main__':
    main()
