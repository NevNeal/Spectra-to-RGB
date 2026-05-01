# ============================================================
# Average Patch RGBs + Spectra RGBs → CSV Summary
# ============================================================

import os, glob, requests
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

# ============================================================
# 1. Paths
# ============================================================

ROOT = Path("/content/drive/MyDrive/spectra_to_rgb/flower_jpgs_2/flower_patches+spectra")
OUTPUT_CSV = ROOT / "patch_spectra_rgb_means.csv"
cmf_path = "/content/drive/MyDrive/spectra_to_rgb/cie-cmf.txt"

if not os.path.exists(cmf_path):
    url = "https://raw.githubusercontent.com/neozhaoliang/pywonderland/master/tools/cie-cmf.txt"
    r = requests.get(url)
    open(cmf_path, "w").write(r.text)
    print("📥 Downloaded cie-cmf.txt")

# ============================================================
# 2. ColourSystem Class (validated spectra → RGB)
# ============================================================

def xyz_from_xy(x, y):
    return np.array((x, y, 1 - x - y))


#defines relationship between spectra and RGB  
class ColourSystem:
    cmf = np.loadtxt(cmf_path, usecols=(1, 2, 3))

    def __init__(self, red, green, blue, white):
        self.red, self.green, self.blue, self.white = red, green, blue, white
        self.M = np.vstack((self.red, self.green, self.blue)).T
        self.MI = np.linalg.inv(self.M)
        self.wscale = self.MI.dot(self.white)
        self.T = self.MI / self.wscale[:, np.newaxis]

    def xyz_to_rgb(self, xyz):
        rgb = self.T.dot(xyz)
        if np.any(rgb < 0):
            rgb += -np.min(rgb)
        if np.max(rgb) > 0:
            rgb /= np.max(rgb)
        rgb = np.where(rgb <= 0.0031308,
                       12.92 * rgb,
                       1.055 * (rgb ** (1 / 2.4)) - 0.055)
        return np.clip(rgb, 0, 1)

    def spec_to_xyz(self, spec):
        XYZ = np.sum(spec[:, np.newaxis] * self.cmf, axis=0)
        den = np.sum(self.cmf[:, 1])
        return XYZ / den if den > 0 else XYZ

    def spec_to_rgb(self, spec):
        return self.xyz_to_rgb(self.spec_to_xyz(spec))

# --- Initialize CIE D65 colour system ---
illuminant_D65 = xyz_from_xy(0.3127, 0.3291)
cs_srgb = ColourSystem(
    red=xyz_from_xy(0.64, 0.33),
    green=xyz_from_xy(0.30, 0.60),
    blue=xyz_from_xy(0.15, 0.06),
    white=illuminant_D65,
)

# ============================================================
# 3. Helper: Convert spectrum file → RGB (0–255)
# ============================================================

def load_spectrum_rgb(file_path):
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
        start_idx = None
        for i, line in enumerate(lines):
            if "Begin Spectral Data" in line:
                start_idx = i + 1
                break
        if start_idx is None:
            raise ValueError("No 'Begin Spectral Data' marker found.")

        data = []
        for line in lines[start_idx:]:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    wl, inten = float(parts[0]), float(parts[1])
                    data.append((wl, inten))
                except:
                    continue

        if not data:
            return None

        df = pd.DataFrame(data, columns=["wavelength", "intensity"])
        df = df[df["wavelength"].between(190, 900)]
        df = df[df["intensity"] > 0].sort_values("wavelength")
        lam = np.arange(380., 781., 5)
        spec = np.interp(lam, df["wavelength"], df["intensity"])
        spec /= np.max(spec)
        rgb = cs_srgb.spec_to_rgb(spec)
        return rgb * 255
    except Exception as e:
        print(f"⚠️ Error reading {os.path.basename(file_path)}: {e}")
        return None

# ============================================================
# 4. Process all subfolders
# ============================================================

records = []

for folder in sorted(ROOT.iterdir()):
    if not folder.is_dir():
        continue

    img_files = list(folder.glob("*.png"))
    spectra_files = list(folder.glob("*.txt"))

    if not img_files or not spectra_files:
        print(f"⚠️ Skipping {folder.name}: missing image or spectra.")
        continue

    image_path = img_files[0]
    print(f"🎨 Processing {folder.name} ({len(spectra_files)} spectra)")

    # --- Average image RGB values ---
    img = Image.open(image_path).convert("RGB")
    img_arr = np.array(img).reshape(-1, 3)
    patch_mean = img_arr.mean(axis=0)

    # --- Average spectra RGB values ---
    spectra_rgb_list = []
    for fp in spectra_files:
        if "cie-cmf" in fp.name.lower():
            continue
        rgb = load_spectrum_rgb(fp)
        if rgb is not None:
            spectra_rgb_list.append(rgb)

    if spectra_rgb_list:
        spectra_mean = np.mean(np.vstack(spectra_rgb_list), axis=0)
    else:
        spectra_mean = np.array([np.nan, np.nan, np.nan])

    # --- Record results ---
    records.append({
        "patch_name": folder.name,
        "patch_R": patch_mean[0],
        "patch_G": patch_mean[1],
        "patch_B": patch_mean[2],
        "spectra_R": spectra_mean[0],
        "spectra_G": spectra_mean[1],
        "spectra_B": spectra_mean[2],
    })

# ============================================================
# 5. Save to CSV
# ============================================================

df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved averages → {OUTPUT_CSV}")
print(df.head())
