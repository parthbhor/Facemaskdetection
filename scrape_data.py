"""
Face Mask Detection System — Web Scraper
=========================================
Scrapes face mask images from Google Images and Bing Images
to build / supplement the training dataset.

Scraped images are saved to:
  D:/python/Facemaskdetect/files/dataset/
    ├── with_mask/        ← people wearing masks
    └── without_mask/     ← people without masks

Requirements:
  pip install requests beautifulsoup4 selenium webdriver-manager Pillow tqdm

Usage:
  python scrape_data.py                        # scrape both classes
  python scrape_data.py --class with_mask      # scrape only one class
  python scrape_data.py --limit 300            # scrape 300 per class
  python scrape_data.py --source bing          # only Bing
  python scrape_data.py --mode add             # add to existing dataset

Note:
  - Uses Selenium (headless Chrome) + requests fallback
  - Validates every image before saving (checks it's a real image)
  - Removes duplicates automatically
  - Resizes nothing — train.py handles preprocessing
"""

import os
import re
import time
import json
import uuid
import random
import hashlib
import argparse
import urllib.request
import urllib.parse
from io import BytesIO
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from PIL import Image
from tqdm import tqdm

# ─── try to import selenium (needed for Google) ───────────────────────────────
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("[WARN] Selenium not found. Google scraping disabled. "
          "Install: pip install selenium webdriver-manager")

# ─── HARDCODED PATHS ──────────────────────────────────────────────────────────
DATASET_PATH = r"D:\python\Facemaskdetect\files\dataset"

# ─── SEARCH QUERIES per class ─────────────────────────────────────────────────
# Multiple queries give more diverse images
SEARCH_QUERIES = {
    "with_mask": [
        "person wearing face mask",
        "people wearing surgical mask",
        "face mask covid",
        "man wearing mask",
        "woman wearing face mask",
        "child wearing mask",
        "N95 mask face",
        "medical face mask person",
        "cloth face mask person",
        "face mask outdoors",
    ],
    "without_mask": [
        "person face no mask",
        "human face portrait",
        "face close up no mask",
        "man face without mask",
        "woman face portrait",
        "person face covid no mask",
        "human face front view",
        "face photo normal",
        "people without mask",
        "face portrait clear",
    ],
}

# ─── USER AGENTS (rotate to avoid blocks) ─────────────────────────────────────
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) "
    "Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

def get_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }

# ─── ARGS ─────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default=DATASET_PATH,
                help="Output dataset directory")
ap.add_argument("--class",   dest="cls", default="both",
                choices=["with_mask", "without_mask", "both"],
                help="Which class to scrape")
ap.add_argument("--limit",   type=int, default=500,
                help="Target images per class (default: 500)")
ap.add_argument("--source",  default="both",
                choices=["google", "bing", "both"],
                help="Image source (default: both)")
ap.add_argument("--mode",    default="add",
                choices=["add", "replace"],
                help="'add' to keep existing images, 'replace' to clear first")
ap.add_argument("--delay",   type=float, default=1.5,
                help="Seconds to wait between requests (default: 1.5)")
args = vars(ap.parse_args())

# ─── SETUP DIRECTORIES ────────────────────────────────────────────────────────
classes_to_scrape = (
    ["with_mask", "without_mask"] if args["cls"] == "both"
    else [args["cls"]]
)

for cls in classes_to_scrape:
    cls_dir = os.path.join(args["dataset"], cls)
    if args["mode"] == "replace" and os.path.exists(cls_dir):
        import shutil
        shutil.rmtree(cls_dir)
        print(f"[INFO] Cleared existing: {cls_dir}")
    os.makedirs(cls_dir, exist_ok=True)

print("=" * 62)
print("  Face Mask Image Scraper")
print("=" * 62)
print(f"  Dataset  : {args['dataset']}")
print(f"  Classes  : {classes_to_scrape}")
print(f"  Target   : {args['limit']} images per class")
print(f"  Sources  : {args['source']}")
print(f"  Mode     : {args['mode']}")
print("=" * 62)

# ─── IMAGE VALIDATION ─────────────────────────────────────────────────────────
def is_valid_image(img_bytes, min_size=50):
    """
    Validates that the bytes represent a real, usable image.
    Checks:
      - Parseable by PIL
      - At least min_size x min_size pixels
      - Not grayscale-only (we need RGB)
      - Not corrupt
    """
    try:
        img = Image.open(BytesIO(img_bytes))
        img.verify()   # checks for corruption
        img = Image.open(BytesIO(img_bytes))   # reopen after verify
        if img.mode not in ("RGB", "RGBA", "L"):
            return False
        w, h = img.size
        if w < min_size or h < min_size:
            return False
        return True
    except Exception:
        return False


def image_hash(img_bytes):
    """MD5 hash of image bytes — used to skip duplicates."""
    return hashlib.md5(img_bytes).hexdigest()


def save_image(img_bytes, save_dir, existing_hashes):
    """
    Saves image to disk if it is valid and not a duplicate.
    Returns True if saved, False otherwise.
    """
    if not is_valid_image(img_bytes):
        return False
    h = image_hash(img_bytes)
    if h in existing_hashes:
        return False   # duplicate
    existing_hashes.add(h)

    ext      = "jpg"
    filename = f"{uuid.uuid4().hex}.{ext}"
    path     = os.path.join(save_dir, filename)

    # Convert to RGB JPEG before saving
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img.save(path, "JPEG", quality=90)
    return True


def download_image(url, timeout=10):
    """Downloads image bytes from a URL. Returns bytes or None."""
    try:
        resp = requests.get(
            url,
            headers=get_headers(),
            timeout=timeout,
            stream=True
        )
        if resp.status_code == 200:
            content_type = resp.headers.get("Content-Type", "")
            if "image" in content_type or content_type == "":
                return resp.content
    except Exception:
        pass
    return None


# ─── BING IMAGE SCRAPER ───────────────────────────────────────────────────────
def scrape_bing(query, limit, save_dir, existing_hashes, delay=1.5):
    """
    Scrapes Bing Images for a given query.
    Uses direct HTTP requests (no browser needed).
    Bing paginates via 'first' offset parameter.
    """
    saved   = 0
    offset  = 0
    seen_urls = set()

    print(f"\n  [Bing] Query: '{query}'")

    while saved < limit:
        encoded_q = urllib.parse.quote(query)
        url = (
            f"https://www.bing.com/images/search"
            f"?q={encoded_q}&first={offset}&count=35"
            f"&safeSearch=Off&form=HDRSC2"
        )

        try:
            resp = requests.get(url, headers=get_headers(), timeout=15)
            if resp.status_code != 200:
                print(f"  [Bing] HTTP {resp.status_code} at offset {offset}")
                break

            soup = BeautifulSoup(resp.text, "html.parser")

            # Bing stores image data in 'murl' inside JSON within <a> tags
            img_urls = []

            # Method 1: parse JSON from data-m attribute
            for tag in soup.find_all("a", {"class": "iusc"}):
                try:
                    data = json.loads(tag.get("m", "{}"))
                    murl = data.get("murl", "")
                    if murl and murl not in seen_urls:
                        img_urls.append(murl)
                        seen_urls.add(murl)
                except Exception:
                    pass

            # Method 2: fallback — look for srcset / src in img tags
            if not img_urls:
                for img_tag in soup.find_all("img", {"class": "mimg"}):
                    src = img_tag.get("src") or img_tag.get("data-src", "")
                    if src.startswith("http") and src not in seen_urls:
                        img_urls.append(src)
                        seen_urls.add(src)

            if not img_urls:
                print(f"  [Bing] No more images found at offset {offset}.")
                break

            for img_url in img_urls:
                if saved >= limit:
                    break
                img_bytes = download_image(img_url)
                if img_bytes and save_image(img_bytes, save_dir, existing_hashes):
                    saved += 1

            offset += 35
            time.sleep(delay + random.uniform(0, 0.5))

        except Exception as e:
            print(f"  [Bing] Error: {e}")
            break

    return saved


# ─── GOOGLE IMAGE SCRAPER (Selenium) ─────────────────────────────────────────
def build_chrome_driver():
    """Builds a headless Chrome WebDriver."""
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(f"--user-agent={random.choice(USER_AGENTS)}")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-gpu")

    service = Service(ChromeDriverManager().install())
    driver  = webdriver.Chrome(service=service, options=options)
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver


def extract_google_img_urls(driver, query, max_scroll=15):
    """
    Opens Google Images for the query, scrolls to load more images,
    and extracts all full-resolution image URLs.
    """
    encoded_q = urllib.parse.quote(query)
    url = f"https://www.google.com/search?q={encoded_q}&tbm=isch&safe=off"

    driver.get(url)
    time.sleep(2)

    img_urls = set()

    for scroll_n in range(max_scroll):
        # Scroll down
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.8)

        # Click "Show more results" button if present
        try:
            more_btn = driver.find_element(By.CSS_SELECTOR, "input.mye4qd")
            if more_btn.is_displayed():
                driver.execute_script("arguments[0].click();", more_btn)
                time.sleep(2)
        except Exception:
            pass

        # Click each thumbnail to get the full-res URL
        thumbnails = driver.find_elements(By.CSS_SELECTOR, "img.Q4LuWd")

        for thumb in thumbnails:
            try:
                thumb.click()
                time.sleep(0.6)
                # Full image appears in a side panel
                full_imgs = driver.find_elements(
                    By.CSS_SELECTOR, "img.n3VNCb, img.iPVvYb, img.r48jcc"
                )
                for fi in full_imgs:
                    src = fi.get_attribute("src") or fi.get_attribute("data-src") or ""
                    if src.startswith("http") and not src.startswith("data:"):
                        img_urls.add(src)
            except Exception:
                pass

        # Also grab URLs from JSON embedded in page source
        page_src = driver.page_source
        raw_urls = re.findall(r'"(https?://[^"]+\.(?:jpg|jpeg|png|webp))"', page_src)
        for u in raw_urls:
            if "gstatic" not in u and "google" not in u:
                img_urls.add(u)

    return list(img_urls)


def scrape_google(query, limit, save_dir, existing_hashes, delay=1.5, driver=None):
    """Scrapes Google Images for a query using Selenium."""
    if not SELENIUM_AVAILABLE:
        print("  [Google] Selenium not available — skipping.")
        return 0

    own_driver = driver is None
    if own_driver:
        try:
            driver = build_chrome_driver()
        except Exception as e:
            print(f"  [Google] ChromeDriver error: {e}")
            return 0

    print(f"\n  [Google] Query: '{query}'")
    saved = 0
    try:
        urls = extract_google_img_urls(driver, query, max_scroll=12)
        for img_url in urls:
            if saved >= limit:
                break
            img_bytes = download_image(img_url)
            if img_bytes and save_image(img_bytes, save_dir, existing_hashes):
                saved += 1
            time.sleep(delay * 0.3)
    except Exception as e:
        print(f"  [Google] Error: {e}")
    finally:
        if own_driver and driver:
            driver.quit()

    return saved


# ─── MAIN SCRAPING PIPELINE ───────────────────────────────────────────────────
def scrape_class(class_name, target_count, source, delay):
    """
    Full pipeline for scraping one class:
      1. Load existing image hashes (dedup against already-downloaded)
      2. For each search query → scrape Bing and/or Google
      3. Stop once target_count is reached
    """
    save_dir = os.path.join(args["dataset"], class_name)
    queries  = SEARCH_QUERIES[class_name]

    # Load hashes of already-downloaded images
    existing_hashes = set()
    existing_count  = 0
    for fname in os.listdir(save_dir):
        fpath = os.path.join(save_dir, fname)
        try:
            with open(fpath, "rb") as f:
                existing_hashes.add(image_hash(f.read()))
            existing_count += 1
        except Exception:
            pass

    print(f"\n{'─'*62}")
    print(f"  CLASS: {class_name}")
    print(f"  Existing images : {existing_count}")
    print(f"  Target total    : {existing_count + target_count}")
    print(f"{'─'*62}")

    total_saved = 0
    per_query   = max(1, target_count // len(queries))

    # Start a shared Selenium driver for Google (reuse across queries)
    google_driver = None
    if (source in ("google", "both")) and SELENIUM_AVAILABLE:
        try:
            print("  [Google] Starting Chrome headless driver...")
            google_driver = build_chrome_driver()
            print("  [Google] Driver ready.")
        except Exception as e:
            print(f"  [Google] Driver failed: {e}")
            google_driver = None

    for i, query in enumerate(queries, 1):
        remaining = target_count - total_saved
        if remaining <= 0:
            break

        q_limit = min(per_query, remaining)
        print(f"\n  Query {i}/{len(queries)}: \"{query}\"  (want {q_limit} images)")

        q_saved = 0

        # ── BING ──
        if source in ("bing", "both"):
            n = scrape_bing(
                query=query,
                limit=q_limit,
                save_dir=save_dir,
                existing_hashes=existing_hashes,
                delay=delay,
            )
            q_saved   += n
            total_saved += n
            print(f"    Bing  → +{n} images saved")

        # ── GOOGLE ──
        if source in ("google", "both") and google_driver:
            g_limit = max(1, q_limit - q_saved)
            n = scrape_google(
                query=query,
                limit=g_limit,
                save_dir=save_dir,
                existing_hashes=existing_hashes,
                delay=delay,
                driver=google_driver,
            )
            q_saved   += n
            total_saved += n
            print(f"    Google → +{n} images saved")

        print(f"  Progress: {total_saved}/{target_count} ({class_name})")
        time.sleep(delay)

    # Cleanup driver
    if google_driver:
        try:
            google_driver.quit()
        except Exception:
            pass

    final_count = len(os.listdir(save_dir))
    print(f"\n  ✓ {class_name} complete:")
    print(f"    Newly saved  : {total_saved}")
    print(f"    Total in dir : {final_count}")
    return total_saved


# ─── RUN ──────────────────────────────────────────────────────────────────────
summary = {}
for cls in classes_to_scrape:
    n = scrape_class(
        class_name   = cls,
        target_count = args["limit"],
        source       = args["source"],
        delay        = args["delay"],
    )
    summary[cls] = n

# ─── FINAL SUMMARY ────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  SCRAPING COMPLETE")
print("=" * 62)
for cls in classes_to_scrape:
    final = len(os.listdir(os.path.join(args["dataset"], cls)))
    print(f"  {cls:20s} → {final:,} total images in dataset")
print()
print(f"  Dataset saved to: {args['dataset']}")
print("  Run training next: python train.py")
print("=" * 62)
