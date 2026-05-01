# ============================================================
# 3D RGB Plot: Image vs Spectra Averages (Color Distance)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Load averaged RGB data ---
csv_path = "/content/drive/MyDrive/spectra_to_rgb/flower_jpgs_2/flower_patches+spectra/patch_spectra_rgb_means.csv"
df = pd.read_csv(csv_path)

# --- Ensure numeric RGBs ---
for col in ["patch_R", "patch_G", "patch_B", "spectra_R", "spectra_G", "spectra_B"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# --- Helper: Extract species label (sp1–sp5) ---
df["species"] = df["patch_name"].str.extract(r"(sp\d+)")
df.dropna(subset=["species"], inplace=True)

# --- Compute Euclidean RGB distances ---
df["RGB_distance"] = np.sqrt(
    (df["patch_R"] - df["spectra_R"]) ** 2 +
    (df["patch_G"] - df["spectra_G"]) ** 2 +
    (df["patch_B"] - df["spectra_B"]) ** 2
)

# --- Marker maps ---
patch_marker = "s"  # all image points are squares
spectra_marker_map = {
    "sp1": "*",  # star
    "sp2": "^",  # triangle
    "sp3": "o",  # circle
    "sp4": "p",  # pentagon
    "sp5": "h",  # hexagon
}

# --- Line color mapping based on dominant RGB difference ---
def dominant_rgb_color(row):
    diffs = np.abs([
        row["patch_R"] - row["spectra_R"],
        row["patch_G"] - row["spectra_G"],
        row["patch_B"] - row["spectra_B"]
    ])
    max_idx = np.argmax(diffs)
    if max_idx == 0:
        return "#FF5555"  # red
    elif max_idx == 1:
        return "#55CC55"  # green
    else:
        return "#5599FF"  # blue

# --- Normalize alpha by distance ---
min_dist, max_dist = df["RGB_distance"].min(), df["RGB_distance"].max()
def distance_alpha(dist):
    # scale between 0.3 (low diff) and 1.0 (high diff)
    if max_dist == min_dist:
        return 1.0
    return 0.3 + 0.7 * ((dist - min_dist) / (max_dist - min_dist))

# --- Initialize 3D figure ---
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

# --- Plot each patch ---
for _, row in df.iterrows():
    species = row["species"]

    patch_rgb = np.array([row["patch_R"], row["patch_G"], row["patch_B"]])
    spectra_rgb = np.array([row["spectra_R"], row["spectra_G"], row["spectra_B"]])

    # --- Normalize RGB for plotting colors ---
    patch_color = patch_rgb / 255.0
    spectra_color = spectra_rgb / 255.0

    line_color = dominant_rgb_color(row)
    alpha_val = distance_alpha(row["RGB_distance"])
    size = 600

    # --- Plot patch (image) point as square (color = patch average) ---
    ax.scatter(
        *patch_rgb,
        color=patch_color,
        marker=patch_marker,
        s=size,
        edgecolors="black",
        linewidths=1.2,
        zorder=3
    )

    # --- Plot spectra point (species-specific marker, color = spectra average) ---
    m = spectra_marker_map.get(species, "o")
    ax.scatter(
        *spectra_rgb,
        color=spectra_color,
        marker=m,
        s=size,
        edgecolors="black",
        linewidths=1.2,
        zorder=3
    )

    # --- Draw line between patch and spectra ---
    ax.plot(
        [patch_rgb[0], spectra_rgb[0]],
        [patch_rgb[1], spectra_rgb[1]],
        [patch_rgb[2], spectra_rgb[2]],
        color=line_color,
        linestyle="-",
        linewidth=3,
        alpha=alpha_val,
        zorder=2
    )

    # --- Midpoint for distance label ---
    mid = (patch_rgb + spectra_rgb) / 2
    ax.text(
        *mid,
        f"{row['RGB_distance']:.1f}",
        color="black",
        fontsize=9,
        ha="center",
        zorder=5
    )

# --- Auto-zoom with 10% margin ---
all_rgb = np.concatenate([
    df[["patch_R", "patch_G", "patch_B"]].values,
    df[["spectra_R", "spectra_G", "spectra_B"]].values
])
min_vals = all_rgb.min(axis=0)
max_vals = all_rgb.max(axis=0)
ranges = max_vals - min_vals

xlim = (min_vals[0] - 0.1 * ranges[0], max_vals[0] + 0.1 * ranges[0])
ylim = (min_vals[1] - 0.1 * ranges[1], max_vals[1] + 0.1 * ranges[1])
zlim = (min_vals[2] - 0.1 * ranges[2], max_vals[2] + 0.1 * ranges[2])

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_zlim(*zlim)

ax.set_xlabel("Red", fontsize=12)
ax.set_ylabel("Green", fontsize=12)
ax.set_zlabel("Blue", fontsize=12)
ax.set_title("Image vs Spectra RGB Averages", pad=35, fontsize=14)

# --- Legends ---
species_legend = [
    plt.Line2D([0], [0], marker="*", color="w",
               label="sp1 – Lantana strigocamara (Common Lantana, Verbenaceae)",
               markerfacecolor="gray", markersize=12, markeredgecolor="black"),
    plt.Line2D([0], [0], marker="^", color="w",
               label="sp2 – Fallugia paradoxa (Apache Plume, Rosaceae)",
               markerfacecolor="gray", markersize=12, markeredgecolor="black"),
    plt.Line2D([0], [0], marker="o", color="w",
               label="sp3 – Encelia farinosa (Brittlebush, Asteraceae)",
               markerfacecolor="gray", markersize=12, markeredgecolor="black"),
    plt.Line2D([0], [0], marker="p", color="w",
               label="sp4 – Tecoma stans (Yellow-bells, Bignoniaceae)",
               markerfacecolor="gray", markersize=12, markeredgecolor="black"),
    plt.Line2D([0], [0], marker="h", color="w",
               label="sp5 – Anisacanthus thurberi (Thurber’s Desert Honeysuckle, Acanthaceae)",
               markerfacecolor="gray", markersize=12, markeredgecolor="black"),
    plt.Line2D([0], [0], marker="s", color="w",
               label="All patches (image averages)",
               markerfacecolor="gray", markersize=12, markeredgecolor="black"),
]

line_legend = [
    plt.Line2D([0], [0], color="#FF5555", lw=3, label="Red difference dominant"),
    plt.Line2D([0], [0], color="#55CC55", lw=3, label="Green difference dominant"),
    plt.Line2D([0], [0], color="#5599FF", lw=3, label="Blue difference dominant"),
]

# Move the legend up slightly using bbox_to_anchor
first_legend = ax.legend(
    handles=species_legend,
    loc="upper left",
    title="Spectra Markers by Species",
    bbox_to_anchor=(0, 1.05)  # move up by roughly 15 pixels
)
second_legend = ax.legend(
    handles=line_legend,
    loc="lower right",
    title="Line Color and Opacity"
)
ax.add_artist(first_legend)



plt.tight_layout()
plt.show()
