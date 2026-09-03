# FTIR PCA Workbench

A PySide6 desktop wrapper for FTIR preprocessing, PCA, K-Means, Ward hierarchical clustering, and DBSCAN.

## Install with Conda

```powershell
conda create -n ftirpca python=3.11 pyside6 numpy pandas scipy scikit-learn matplotlib -y
conda activate ftirpca
python ftir_pca_gui.py
```

Or, from an existing compatible environment:

```powershell
conda install pyside6 numpy pandas scipy scikit-learn matplotlib -y
python ftir_pca_gui.py
```

## Dataset

The loader supports the supplied layout where the first data row contains wavenumbers and subsequent rows contain sample metadata and intensities. It also supports CSVs whose spectral column names are numeric wavenumbers.

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

## Interpretation caution

Clusters are exploratory mathematical groupings, not confirmed chemical identities. Validate separation using loadings, raw/preprocessed spectra, replicates, known controls, and chemical context.
