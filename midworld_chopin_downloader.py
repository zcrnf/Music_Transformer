import os
import requests
from bs4 import BeautifulSoup

# URL to scrape
BASE_URL = "https://www.midiworld.com/chopin.htm"
SAVE_DIR = "midi_files"

os.makedirs(SAVE_DIR, exist_ok=True)

def scrape_midi_links():
    resp = requests.get(BASE_URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    midi_links = []
    for link in soup.find_all("a", href=True):
        href = link['href']
        if href.endswith(".mid") or href.endswith(".MID"):
            full_url = href if href.startswith("http") else f"https://www.midiworld.com/{href}"
            midi_links.append(full_url)
    return midi_links

def download_midis(midi_links):
    for url in midi_links:
        filename = url.split("/")[-1]
        save_path = os.path.join(SAVE_DIR, filename)
        if os.path.exists(save_path):
            print(f"⏩ Already downloaded: {filename}")
            continue
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            with open(save_path, "wb") as f:
                f.write(r.content)
            print(f"✅ Downloaded: {filename}")
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")

if __name__ == "__main__":
    print("🚀 Scraping MIDIWorld Chopin page...")
    links = scrape_midi_links()
    print(f"🎵 Found {len(links)} MIDI files.")
    download_midis(links)
    print("🎉 All available MIDI files downloaded!")






import requests
from bs4 import BeautifulSoup
from pathlib import Path

# === Configuration ===
BASE_URL = "https://www.classicalmidi.co.uk/chopin.htm"
SAVE_DIR = Path("midi_files")  # <- Save into your old midi_files folder
SAVE_DIR.mkdir(exist_ok=True)

# Keywords for pieces you want (lowercase match)
must_download_keywords = [
    "etude", "winter wind", "valse", "prelude", "nocturne",
    "impromptu", "sonata", "polish song", "fantasy", "scherzo",
    "barcarolle", "variations", "ecossaise", "finale to b minor"
]

# Helper: Normalize text
def normalize(text):
    return text.lower().replace('-', ' ').replace('_', ' ')

# === Step 1: Scrape the page ===
resp = requests.get(BASE_URL)
resp.raise_for_status()
soup = BeautifulSoup(resp.text, "html.parser")

# === Step 2: Find all download links matching needed keywords ===
download_targets = []

for link in soup.find_all("a", href=True):
    href = link["href"]
    text = link.get_text().strip()

    if href.endswith(".mid") and any(keyword in normalize(text) for keyword in must_download_keywords):
        full_url = href
        if not full_url.startswith("http"):
            full_url = "https://www.classicalmidi.co.uk/" + full_url.lstrip('/')
        download_targets.append((text, full_url))

# === Step 3: Download missing files ===
for name, url in download_targets:
    filename = name.replace(" ", "_")[:80] + ".mid"  # limit filename length a bit
    save_path = SAVE_DIR / filename

    if save_path.exists():
        print(f"⏩ Already downloaded: {filename}")
        continue

    try:
        print(f"⬇️ Downloading: {filename} ...")
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(r.content)
        print(f"✅ Saved: {filename}")
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")

print(f"🎉 Done! Downloaded {len(download_targets)} required Chopin MIDIs.")
