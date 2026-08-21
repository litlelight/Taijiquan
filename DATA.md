# Data required to reproduce the analysis

The dataset is **not redistributed in this repository**. Download it from the original UMONS-TAICHI sources. The analysis uses TSV and Kinect files; C3D and video files are not required.

## 1. Metadata

Clone the official dataset repository so `Metadata.txt` is available at `data/UMONS-TAICHI/Metadata.txt`:

```bash
git clone --depth 1 https://github.com/numediart/UMONS-TAICHI.git data/UMONS-TAICHI
```

Official repository: https://github.com/numediart/UMONS-TAICHI

## 2. Motion-capture archives

Official Zenodo record: https://zenodo.org/records/2784581

Place the following files directly under `data/`:

| Required file | Official download | Zenodo MD5 |
|---|---|---|
| `Labels.zip` | https://zenodo.org/records/2784581/files/Labels.zip?download=1 | `35de96a035c9d3b41619ac1f9932f946` |
| `Kinect.zip` | https://zenodo.org/records/2784581/files/Kinect.zip?download=1 | `d2802095beec3736ecfb35ab4eac70e8` |
| `Segmented_Kinect.zip` | https://zenodo.org/records/2784581/files/Segmented_Kinect.zip?download=1 | `f783bbe1c8c19dbebb9ca3dbc2265f83` |
| `TSV.zip` | https://zenodo.org/records/2784581/files/TSV.zip?download=1 | `904272e5599f51cb473ce8396c3bc4df` |
| `Segmented_TSV.zip` | https://zenodo.org/records/2784581/files/Segmented_TSV.zip?download=1 | `03969cc686f0a750437069c116ff4e21` |

After downloading, run:

```bash
python check_data.py
```

The data check validates the five archive MD5 hashes, confirms that `Metadata.txt` contains 12 participants, and verifies the released-file structure used by the revised analysis (131 labels/raw Qualisys recordings, 111 raw Kinect recordings, 2,149 segmented Qualisys files, and 1,815 segmented Kinect files).

The UMONS-TAICHI dataset is described in Tits et al., *Data in Brief* (2018), DOI: 10.1016/j.dib.2018.05.088.
