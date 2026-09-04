# FTIR PCA Workbench

A PySide6 desktop wrapper for FTIR preprocessing, PCA, K-Means, Ward hierarchical clustering, and DBSCAN.

Preprocessing (SNV, Savitzky-Golay smoothing/derivatives) and PCA are performed with
[SpectroChemPy](https://www.spectrochempy.fr/). Clustering (KMeans/Agglomerative/DBSCAN),
which SpectroChemPy does not provide, still uses scikit-learn on the resulting PCA scores.

## Install with Conda

```powershell
conda create -n ftirpca python=3.11 pyside6 numpy pandas scipy scikit-learn matplotlib -y
conda activate ftirpca
pip install spectrochempy mplcursors
python ftir_pca_gui.py
```

Or, from an existing compatible environment:

```powershell
conda install pyside6 numpy pandas scipy scikit-learn matplotlib -y
pip install spectrochempy mplcursors
python ftir_pca_gui.py
```

`mplcursors` is optional — it enables hover tooltips on the plots. Without it, the
plots still work (pan/zoom/save toolbar and click-to-select in the scores table),
just without hover labels.

## Dataset

Two input modes are available in the UI:

- **Wide-format CSV** — the original loader, supporting the supplied layout where the
  first data row contains wavenumbers and subsequent rows contain sample metadata and
  intensities. It also supports CSVs whose spectral column names are numeric wavenumbers.
- **Individual .spa files (OMNIC)** — select one or more Thermo/Nicolet OMNIC `.spa`
  files directly (e.g. one file per sample) via "Add .spa files…". They are read and
  merged into a single dataset with SpectroChemPy's OMNIC reader
  (`spectrochempy.read_omnic`). All selected files must share the same wavenumber axis
  (same instrument range/resolution) to be merged for PCA.

## Recommended starting settings

- Preprocessing: First derivative + SNV
- Analysis range: 400 to 4000 cm-1
- Excluded ranges: 500-1000
- SG window: 21
- SG polynomial: 3
- PCA components: 30
- Clustering PCs: 10

Multiple excluded regions can be entered as `500-1000, 1800-1900`.

## Outputs

Each run saves:

- `pca_scores_and_clusters.csv`
- `pca_loadings.csv`
- `kmeans_silhouette.csv`
- `analysis_summary.json`

The UI provides tabs for spectra, PCA scores, loadings, variance, clusters, dendrogram, and a score table.

## Interactivity

Every plot tab has a matplotlib navigation toolbar for pan/zoom/box-zoom and saving
an image. On the spectra, PCA scores, and clusters plots you can additionally:

- Hover over a line/point to see its sample ID (requires `mplcursors`).
- Click a point in the PCA scores or clusters plots to jump to and highlight the
  corresponding row in the Scores table.

## Interpretation caution

Clusters are exploratory mathematical groupings, not confirmed chemical identities. Validate separation using loadings, raw/preprocessed spectra, replicates, known controls, and chemical context.
