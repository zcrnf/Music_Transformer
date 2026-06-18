"""
Generate metadata JSONL for MIDI token training
Maps token files to composer IDs for conditioning
"""

import json
from pathlib import Path
from collections import defaultdict
import torch

# Paths
TOKEN_DIR = Path("encoded_tokens/midis")
OUTPUT_FILE = "metadata_clean_midis.jsonl"
MIN_PIECES_PER_COMPOSER = 10  # Minimum pieces to keep composer separate

def get_composer_mapping(token_dir):
    """Generate composer ID mapping with bucketing for rare composers"""
    # Count files per composer
    composer_counts = defaultdict(int)
    for token_file in token_dir.rglob("*.th"):
        composer = token_file.parent.name
        composer_counts[composer] += 1
    
    # Separate frequent and rare composers
    frequent_composers = []
    rare_count = 0
    for composer, count in composer_counts.items():
        if count >= MIN_PIECES_PER_COMPOSER:
            frequent_composers.append(composer)
        else:
            rare_count += count
    
    # Sort frequent composers
    frequent_composers = sorted(frequent_composers)
    
    # Build mapping: 0=unconditional (reserved), 1..K=frequent composers, K+1=OTHER
    # Note: unconditional (0) is handled by CFG dropout in training, not metadata
    composer_to_id = {}
    for idx, composer in enumerate(frequent_composers):
        composer_to_id[composer] = idx + 1  # Start from 1, leaving 0 for unconditional
    
    # All rare composers map to OTHER (last ID)
    other_id = len(frequent_composers) + 1
    for composer in composer_counts:
        if composer not in composer_to_id:
            composer_to_id[composer] = other_id
    
    print(f"\n📊 Composer statistics:")
    print(f"  Total composers: {len(composer_counts)}")
    print(f"  Kept as separate IDs (≥{MIN_PIECES_PER_COMPOSER} pieces): {len(frequent_composers)}")
    print(f"  Mapped to OTHER: {len(composer_counts) - len(frequent_composers)} ({rare_count} files)")
    print(f"  Total composer IDs: {other_id + 1} (0=unconditional, 1-{len(frequent_composers)}=composers, {other_id}=OTHER)")
    print(f"\nTop frequent composers:")
    for composer in frequent_composers[:10]:
        print(f"  {composer}: {composer_counts[composer]} files (ID={composer_to_id[composer]})")
    
    return composer_to_id, frequent_composers, other_id

def create_metadata(token_dir, composer_map):
    """Create metadata JSONL from tokenized files"""
    metadata = []
    stats = defaultdict(int)
    
    for token_file in token_dir.rglob("*.th"):
        composer = token_file.parent.name
        composer_id = composer_map[composer]
        
        # Load to get sequence length
        tokens = torch.load(token_file)
        seq_len = len(tokens)
        
        metadata.append({
            "token_path": str(token_file),
            "composer": composer,
            "composer_id": composer_id,
            "sequence_length": seq_len
        })
        
        stats[composer] += 1
    
    print(f"\n✅ Processed {len(metadata)} token files")
    print(f"\nTop composers by file count:")
    for composer, count in sorted(stats.items(), key=lambda x: -x[1])[:10]:
        print(f"  {composer}: {count} files")
    
    return metadata

if __name__ == "__main__":
    print("🎵 Generating MIDI metadata with composer bucketing...")
    
    # Create composer mapping with bucketing
    composer_map, frequent_composers, other_id = get_composer_mapping(TOKEN_DIR)
    
    # Save composer mapping for reference
    mapping_info = {
        "composer_to_id": composer_map,
        "frequent_composers": frequent_composers,
        "other_id": other_id,
        "min_pieces_threshold": MIN_PIECES_PER_COMPOSER,
        "total_ids": other_id + 1,
        "note": "ID 0 is reserved for unconditional (CFG dropout)"
    }
    with open("composer_mapping.json", "w") as f:
        json.dump(mapping_info, f, indent=2)
    print(f"\n💾 Saved composer mapping to composer_mapping.json")
    
    # Create metadata
    metadata = create_metadata(TOKEN_DIR, composer_map)
    
    # Save as JSONL
    with open(OUTPUT_FILE, "w") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")
    
    print(f"\n💾 Saved {len(metadata)} entries to {OUTPUT_FILE}")
    
    # Print statistics
    lengths = [entry["sequence_length"] for entry in metadata]
    print(f"\n📏 Sequence length statistics:")
    print(f"  Min: {min(lengths)} tokens")
    print(f"  Max: {max(lengths)} tokens")
    print(f"  Median: {sorted(lengths)[len(lengths)//2]} tokens")
    print(f"  Mean: {sum(lengths)//len(lengths)} tokens")
