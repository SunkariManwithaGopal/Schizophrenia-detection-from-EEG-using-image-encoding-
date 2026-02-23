
# Schizophrenia Detection from EEG (Image-encoding)

**Project:** EEG-based schizophrenia detection using image-encoding of EEG signals and deep-feature extraction.

**Summary:**
- **Purpose:** Convert multichannel EEG recordings (MSU dataset) into time-frequency images (scalograms), extract features using pretrained CNNs (EfficientNet / DenseNet), then classify subjects (healthy vs schizophrenia) with an SVM.
- **Input data:** MSU dataset `.eea` files (healthy: `msu_data/norm`, schizo: `msu_data/sch`).

**Repository structure**
- **`SCZ_detection_from_EEG_using_image_encoding_.ipynb`**: Main Jupyter notebook — full pipeline (download, preprocessing, scalogram generation, feature extraction, classification).
- **`msu_data/`**: expected dataset folder with `norm/` and `sch/` subfolders of `.eea` files.
- **`features.npy`**: saved combined features (EfficientNet + DenseNet) produced by the notebook.
- **`README.md`**: this file.

**Requirements**
- Python 3.8+ and the following packages (used in the notebook):
	- `numpy`, `mne`, `scipy`, `pywt`, `scikit-image`, `matplotlib`, `seaborn`
	- `tensorflow` (for EfficientNet), `torch` and `torchvision` (for DenseNet)
	- `scikit-learn`, `joblib`

Install quickly (example):

```powershell
pip install numpy mne scipy pywt scikit-image matplotlib seaborn tensorflow torch torchvision scikit-learn joblib
```

**How to run**
- Open the notebook: [SCZ_detection_from_EEG_using_image_encoding_.ipynb](SCZ_detection_from_EEG_using_image_encoding_.ipynb)
- Step through cells sequentially. Key stages in the notebook:
	- Download / extract MSU dataset (cells include `download_and_unzip` calls)
	- Load `.eea` files into numpy arrays (loader handles 16 channels, 7680 samples per channel)
	- Preprocess: bandpass filter (0.5–45 Hz), re-reference with MNE average reference
	- Generate scalograms per channel (time-frequency images)
	- Extract features on-the-fly using pretrained models (`EfficientNetB3` and a DenseNet backbone)
	- Concatenate features and save as `features.npy`
	- Perform feature selection and classification (SVM + GridSearchCV)

**Notes & tips**
- The notebook is designed to avoid excessive RAM usage by generating scalograms and extracting CNN features on-the-fly, then discarding images.
- GPU is recommended for CNN feature extraction (TensorFlow / PyTorch). If no GPU is available, processing will be slower.
- If you already have `features.npy`, you can skip image generation and run the classification cell directly.
- The loader assumes `.eea` files are plain numeric text matching 16*7680 samples; adjust `load_subject()` if your files differ.

**Outputs**
- `features.npy`: combined feature matrix saved by the notebook.
- Trained SVM model (not auto-saved in the notebook — you may save with `joblib.dump` if desired).

**Next steps / improvements**
- Add CLI or Python scripts to parallelize feature extraction safely.
- Add versioned model checkpoint saving and evaluation scripts.
- Add unit tests for file loading and scalogram generation.

**Contact / Author**
- Maintainer: see notebook metadata or project owner.

