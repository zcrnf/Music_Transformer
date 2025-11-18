#!/usr/bin/env python3
"""
Organize MIDI files by composer name.
Extracts composer from filename (text before first comma) and moves to subfolders.

Example:
    "Chopin, Frederic, Nocturne Op.9 No.2.mid" → data_raw/midis/Chopin/
    "Liszt, Franz, Hungarian Rhapsody.mid" → data_raw/midis/Liszt/
"""

import shutil
from pathlib import Path
from collections import Counter

# Configuration
SOURCE_DIR = Path("data_raw/midis")
OUTPUT_BASE = Path("data_raw/midis_organized")

def extract_composer(filename: str) -> str:
    """Extract composer name from filename (text before first comma)."""
    stem = Path(filename).stem  # Remove .mid extension
    
    # Split by comma and take first part
    if ',' in stem:
        composer = stem.split(',')[0].strip()
        # Clean up common issues
        composer = composer.replace('_', ' ')
        composer = ' '.join(composer.split())  # Normalize whitespace
        return composer
    else:
        # No comma - use whole name as composer
        return "Unknown"

def organize_midis():
    """Organize all MIDI files by composer into subdirectories."""
    
    # Get all MIDI files
    midi_files = list(SOURCE_DIR.glob("*.mid"))
    
    if not midi_files:
        print(f"❌ No MIDI files found in {SOURCE_DIR}")
        return
    
    print(f"🎵 Found {len(midi_files)} MIDI files")
    print(f"📁 Organizing into: {OUTPUT_BASE}\n")
    
    # Count composers
    composer_counts = Counter()
    
    # First pass: count and display
    for midi_file in midi_files:
        composer = extract_composer(midi_file.name)
        composer_counts[composer] += 1
    
    # Show top composers
    print("Top 20 composers found:")
    for composer, count in composer_counts.most_common(20):
        print(f"  {composer:30s}: {count:4d} files")
    
    print(f"\nTotal unique composers: {len(composer_counts)}")
    
    # Ask for confirmation
    proceed = input("\nProceed with organization? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Cancelled.")
        return
    
    # Create output directory
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    
    # Second pass: organize files
    organized_count = 0
    error_count = 0
    
    for midi_file in midi_files:
        composer = extract_composer(midi_file.name)
        
        # Create composer directory
        composer_dir = OUTPUT_BASE / composer
        composer_dir.mkdir(parents=True, exist_ok=True)
        
        # Move file
        dest_path = composer_dir / midi_file.name
        
        try:
            shutil.copy2(midi_file, dest_path)
            organized_count += 1
            
            if organized_count % 500 == 0:
                print(f"  Processed {organized_count}/{len(midi_files)}...")
                
        except Exception as e:
            print(f"❌ Error copying {midi_file.name}: {e}")
            error_count += 1
    
    print(f"\n✅ Organization complete!")
    print(f"   Successfully organized: {organized_count} files")
    print(f"   Errors: {error_count}")
    print(f"   Output directory: {OUTPUT_BASE.resolve()}")
    
    # Show final statistics
    print(f"\n📊 Final composer distribution:")
    for composer, count in composer_counts.most_common(10):
        print(f"   {composer:30s}: {count:4d} files")

if __name__ == "__main__":
    organize_midis()
