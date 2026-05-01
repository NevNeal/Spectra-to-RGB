# ============================================================
# Convert all spectra (.txt) in flower_jpgs_2/organized → RGB colours
# These are all in the 'spectra+RGB figures' folder
# ============================================================

import os, glob, requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# 1. Root folder — new organized structure
# ============================================================
spectra_root = "/content/drive/MyDrive/spectra_to_rgb/flower_jpgs_2/organized"

# ============================================================
# 2. Ensure cie-cmf.txt exists (download if missing)
# ============================================================
cmf_path = "/content/drive/MyDrive/spectra_to_rgb/cie-cmf.txt"
if not os.path.exists(cmf_path):
    url = "https://raw.githubusercontent.com/neozhaoliang/pywonderland/master/tools/cie-cmf.txt"
    r = requests.get(url)
    open(cmf_path, "w").write(r.text)
    print("📥 Downloaded cie-cmf.txt")

# ============================================================
# 3. ColourSystem class (CIE-based conversion, gamma-corrected)
# ============================================================

def xyz_from_xy(x, y):
    """Return the XYZ vector from chromaticity coordinates."""
    return np.array((x, y, 1 - x - y))

class ColourSystem:
    """CIE-based colour conversion with gamma correction."""
    cmf = np.loadtxt(cmf_path, usecols=(1, 2, 3))

    def __init__(self, red, green, blue, white):
        self.red, self.green, self.blue, self.white = red, green, blue, white
        self.M = np.vstack((self.red, self.green, self.blue)).T
        self.MI = np.linalg.inv(self.M)
        self.wscale = self.MI.dot(self.white)
        self.T = self.MI / self.wscale[:, np.newaxis]

    def xyz_to_rgb(self, xyz, out_fmt=None):
        """Convert XYZ → display-ready RGB (gamma-corrected)."""
        rgb = self.T.dot(xyz)
        if np.any(rgb < 0):
            rgb += -np.min(rgb)
        if np.max(rgb) > 0:
            rgb /= np.max(rgb)

        # --- Apply gamma correction for sRGB ---
        rgb = np.where(
            rgb <= 0.0031308,
            12.92 * rgb,
            1.055 * (rgb ** (1 / 2.4)) - 0.055,
        )
        rgb = np.clip(rgb, 0, 1)
        if out_fmt == "html":
            return self.rgb_to_hex(rgb)
        return rgb

    def rgb_to_hex(self, rgb):
        """Convert RGB array to HTML hex string."""
        hex_rgb = (255 * rgb).astype(int)
        return "#%02x%02x%02x" % tuple(hex_rgb)

    def spec_to_xyz(self, spec):
        """Convert a spectrum (380–780 nm, 5 nm steps) to XYZ."""
        XYZ = np.sum(spec[:, np.newaxis] * self.cmf, axis=0)
        den = np.sum(self.cmf[:, 1])
        return XYZ / den if den > 0 else XYZ

    def spec_to_rgb(self, spec, out_fmt=None):
        """Convert a spectrum directly to RGB."""
        return self.xyz_to_rgb(self.spec_to_xyz(spec), out_fmt)

# Instantiate color system
illuminant_D65 = xyz_from_xy(0.3127, 0.3291)
cs_srgb = ColourSystem(
    red=xyz_from_xy(0.64, 0.33),
    green=xyz_from_xy(0.30, 0.60),
    blue=xyz_from_xy(0.15, 0.06),
    white=illuminant_D65,
)

# ============================================================
# 4. Helper: Load & process one spectrum file
# ============================================================

def load_spectrum(file_path):
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
            raise ValueError("No valid numeric rows found in file.")

        df = pd.DataFrame(data, columns=["wavelength", "intensity"])
        df = df[df["wavelength"].between(190, 900)]
        df = df[df["intensity"] > 0].sort_values("wavelength")

        lam = np.arange(380., 781., 5)
        spec = np.interp(lam, df["wavelength"], df["intensity"])
        spec /= np.max(spec)

        rgb = cs_srgb.spec_to_rgb(spec)
        hex_color = cs_srgb.spec_to_rgb(spec, out_fmt="html")
        lam_max = df.loc[df["intensity"].idxmax(), "wavelength"]
        return df, rgb, hex_color, lam_max

    except Exception as e:
        print(f"⚠️ Error reading {os.path.basename(file_path)}: {e}")
        return None, None, None, None

# ============================================================
# 5. Recursively process spectra in all subfolders
# ============================================================

records = []
spectra_files = glob.glob(os.path.join(spectra_root, "**", "*.txt"), recursive=True)
print(f"🔍 Found {len(spectra_files)} spectra files")

for fp in spectra_files:
    if os.path.basename(fp).lower() == "cie-cmf.txt":
        continue
    df, rgb, hex_color, lam_max = load_spectrum(fp)
    if rgb is not None:
        records.append({
            "file_path": os.path.abspath(fp),
            "rgb_array": rgb,
            "hex_color": hex_color,
            "lambda_max_nm": lam_max
        })

spectra_df = pd.DataFrame(records)
print("\n✅ Processed Spectra Summary:")
display(spectra_df)

# ============================================================
# 6. Visualization: save graph in same folder as spectrum
# ============================================================

def visualize_spectra_in_place(spectra_df):
    if spectra_df.empty:
        print("⚠️ No spectra data to visualize.")
        return

    rgb_refs = {"R": 700, "G": 546, "B": 435}
    rgb_colors = {"R": "red", "G": "lime", "B": "blue"}

    for i, row in enumerate(spectra_df.itertuples()):
        file_name = os.path.basename(row.file_path)
        folder = os.path.dirname(row.file_path)
        out_path = os.path.join(folder, file_name.replace(".txt", ".png"))

        with open(row.file_path, "r") as f:
            lines = f.readlines()
        start_idx = [i for i, l in enumerate(lines) if "Begin Spectral Data" in l]
        if start_idx:
            start_idx = start_idx[0] + 1
            wl, inten = [], []
            for line in lines[start_idx:]:
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        wl.append(float(parts[0]))
                        inten.append(float(parts[1]))
                    except:
                        continue
            wl, inten = np.array(wl), np.array(inten)
            inten = np.clip(inten, a_min=0, a_max=None)
            if np.max(inten) > 0:
                inten = inten / np.max(inten)

        fig = plt.figure(figsize=(8, 3))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])

        ax_color = fig.add_subplot(gs[0])
        ax_color.add_patch(plt.Rectangle((0, 0.2), 1, 0.8, color=row.hex_color))
        ax_color.text(0.5, 0.05, row.hex_color, ha='center', va='center',
                      fontsize=12, color='black', weight='bold')
        ax_color.set_xlim(0, 1)
        ax_color.set_ylim(0, 1)
        ax_color.axis("off")

        ax_spec = fig.add_subplot(gs[1])
        ax_spec.plot(wl, inten, color=row.hex_color, lw=2)

        for label, wavelength in rgb_refs.items():
            ax_spec.axvline(wavelength, color=rgb_colors[label],
                            linestyle='--', alpha=0.25, lw=1.5)
            ax_spec.text(wavelength, 1.02, label,
                         color=rgb_colors[label], ha='center',
                         va='bottom', alpha=0.4, fontsize=9, fontweight='bold')

        ax_spec.set_xlim(380, 780)
        ax_spec.set_ylim(0, 1.05)
        ax_spec.set_xlabel("Wavelength (nm)")
        ax_spec.set_ylabel("Normalized Intensity")
        ax_spec.set_title(file_name)
        ax_spec.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"🎨 Saved → {out_path}")

# ============================================================
# 7. Run visualization
# ============================================================

visualize_spectra_in_place(spectra_df)

# ============================================================
# 8. Save RGB summary
# ============================================================

out_csv = os.path.join(spectra_root, "spectra_rgb_summary.csv")
spectra_df.to_csv(out_csv, index=False)
print(f"💾 Saved summary → {out_csv}")
