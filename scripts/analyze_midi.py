#!/usr/bin/env python3
"""Quick script to analyze MIDI file characteristics"""

from pathlib import Path
from miditoolkit import MidiFile
import statistics

MIDI_DIR = Path("data_raw/midis_organized")
midi_files = sorted(list(MIDI_DIR.rglob("*.mid"))[:100])  # Sample first 100

durations = []
note_counts = []
instrument_counts = []

print(f"Analyzing {len(midi_files)} sample MIDI files...\n")

for mf in midi_files:
    try:
        midi = MidiFile(mf)
        duration_sec = midi.max_tick / midi.ticks_per_beat / 2  # Rough estimate
        num_notes = sum(len(inst.notes) for inst in midi.instruments)
        num_instruments = len(midi.instruments)
        
        durations.append(duration_sec)
        note_counts.append(num_notes)
        instrument_counts.append(num_instruments)
    except Exception as e:
        print(f"Error: {mf.name}: {e}")

print("="*60)
print("MIDI DATASET STATISTICS")
print("="*60)
print(f"Sample size: {len(durations)} files")
print()
print("Duration (seconds):")
print(f"  Min:     {min(durations):.1f}s")
print(f"  Max:     {max(durations):.1f}s")
print(f"  Mean:    {statistics.mean(durations):.1f}s")
print(f"  Median:  {statistics.median(durations):.1f}s")
print()
print("Notes per file:")
print(f"  Min:     {min(note_counts)}")
print(f"  Max:     {max(note_counts)}")
print(f"  Mean:    {statistics.mean(note_counts):.0f}")
print(f"  Median:  {statistics.median(note_counts):.0f}")
print()
print("Instruments per file:")
print(f"  Min:     {min(instrument_counts)}")
print(f"  Max:     {max(instrument_counts)}")
print(f"  Mean:    {statistics.mean(instrument_counts):.1f}")
print(f"  Median:  {statistics.median(instrument_counts):.0f}")
print("="*60)

# Show a few examples
print("\nFirst 5 files:")
for i, mf in enumerate(midi_files[:5]):
    print(f"  {mf.name[:50]:50s} {durations[i]:6.1f}s, {note_counts[i]:5d} notes")
