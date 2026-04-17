#!/usr/bin/env python3
"""Download Latin-subset WOFF2 fonts and emit fonts.css."""

import re
import sys
import urllib.request

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

FONT_REQUESTS = [
    # (family_param, css_family, style, weights, filename_prefix)
    (
        "Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;0,8..60,700;1,8..60,400",
        "Source Serif 4",
        ["normal-400", "normal-600", "normal-700", "italic-400"],
    ),
    (
        "Inter:wght@400;500;600",
        "Inter",
        ["normal-400", "normal-500", "normal-600"],
    ),
    (
        "JetBrains+Mono:wght@400;500",
        "JetBrains Mono",
        ["normal-400", "normal-500"],
    ),
]

FONTS_DIR = "src/evidenceengine/static/fonts"
CSS_OUT = "src/evidenceengine/static/css/fonts.css"


def fetch_css(family_param: str) -> str:
    url = f"https://fonts.googleapis.com/css2?family={family_param}&display=swap&subset=latin"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode()


def extract_latin_blocks(css: str) -> list[dict]:
    """Extract only /* latin */ @font-face blocks."""
    blocks = []
    # Split on comment markers
    parts = re.split(r"(/\*[^*]*\*/)", css)
    label = ""
    for part in parts:
        m = re.match(r"/\*\s*(.+?)\s*\*/", part.strip())
        if m:
            label = m.group(1)
        elif "@font-face" in part and label in ("latin", "latin-ext"):
            woff_url = re.search(r"url\((https://fonts\.gstatic\.com[^)]+\.woff2)\)", part)
            style_m = re.search(r"font-style:\s*(\w+)", part)
            weight_m = re.search(r"font-weight:\s*([\w ]+)", part)
            unicode_m = re.search(r"unicode-range:\s*([^;]+);", part)
            family_m = re.search(r"font-family:\s*'([^']+)'", part)
            if woff_url and style_m and weight_m and family_m:
                blocks.append({
                    "family": family_m.group(1),
                    "style": style_m.group(1),
                    "weight": weight_m.group(1).strip(),
                    "url": woff_url.group(1),
                    "unicode": unicode_m.group(1).strip() if unicode_m else None,
                    "subset": label,
                })
    return blocks


def download_font(url: str, dest: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    with open(dest, "wb") as f:
        f.write(data)
    print(f"  Downloaded {dest} ({len(data):,} bytes)")


def family_slug(name: str) -> str:
    return name.lower().replace(" ", "-")


def main() -> None:
    import os
    os.makedirs(FONTS_DIR, exist_ok=True)

    all_css_blocks: list[str] = []

    for family_param, css_family, _ in FONT_REQUESTS:
        print(f"\nFetching CSS for {css_family}...")
        css = fetch_css(family_param)
        blocks = extract_latin_blocks(css)

        # Deduplicate by (family, style, weight, subset)
        seen = set()
        for b in blocks:
            key = (b["family"], b["style"], b["weight"], b["subset"])
            if key in seen:
                continue
            seen.add(key)

            slug = family_slug(b["family"])
            # e.g. source-serif-4-normal-400-latin.woff2
            filename = f"{slug}-{b['style']}-{b['weight'].replace(' ', '')}-{b['subset']}.woff2"
            dest = f"{FONTS_DIR}/{filename}"

            if not os.path.exists(dest):
                print(f"  Downloading {b['family']} {b['style']} {b['weight']} ({b['subset']})...")
                download_font(b["url"], dest)
            else:
                print(f"  Already have {filename}")

            face = f"""@font-face {{
  font-family: '{b["family"]}';
  font-style: {b["style"]};
  font-weight: {b["weight"]};
  font-display: swap;
  src: url('/static/fonts/{filename}') format('woff2');"""
            if b["unicode"]:
                face += f"\n  unicode-range: {b['unicode']};"
            face += "\n}"
            all_css_blocks.append(face)

    css_content = "/* Self-hosted WOFF2 fonts — Latin + Latin-ext subsets */\n\n"
    css_content += "\n\n".join(all_css_blocks) + "\n"

    os.makedirs(os.path.dirname(CSS_OUT), exist_ok=True)
    with open(CSS_OUT, "w") as f:
        f.write(css_content)
    print(f"\nWrote {CSS_OUT}")


if __name__ == "__main__":
    main()
