#!/usr/bin/env python3
"""PySide6 desktop UI for reusable FTIR PCA and clustering analysis.

Preprocessing and PCA are performed with SpectroChemPy (NDDataset, SNV,
Savitzky-Golay derivatives, PCA). Clustering (KMeans/Agglomerative/DBSCAN),
which SpectroChemPy does not provide, still uses scikit-learn on the
SpectroChemPy PCA scores. Individual Thermo/Nicolet OMNIC `.spa` files can be
loaded directly (in addition to the original wide-format CSV layout) via
SpectroChemPy's OMNIC reader.
"""
from __future__ import annotations
import json, re, sys, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

try:
    import spectrochempy as scp
    from spectrochempy import NDDataset, Coord
    SCP_IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover - reported at runtime in the UI
    scp = None; NDDataset = None; Coord = None; SCP_IMPORT_ERROR = str(_e)

try:
    import mplcursors
except ImportError:
    mplcursors = None

from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QGridLayout,
 QGroupBox,QLabel,QLineEdit,QPushButton,QFileDialog,QComboBox,QSpinBox,QDoubleSpinBox,QCheckBox,
 QTabWidget,QPlainTextEdit,QTableWidget,QTableWidgetItem,QMessageBox,QProgressBar,QSplitter,
 QListWidget,QListWidgetItem,QStackedWidget,QRadioButton,QButtonGroup,QAbstractItemView)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

META_DEFAULT=["SampleID","Brand","Grade","City","State","Year"]
SAMPLE_ID_YEAR_RE=re.compile(r"^\s*(\d{4})\.\d+")

def parse_year_from_sample_id(sample_id):
    """Extract the yyyy year from a SampleID formatted as yyyy.nnnn (e.g. 2024.0007)."""
    m=SAMPLE_ID_YEAR_RE.match(str(sample_id))
    return m.group(1) if m else ""

def load_metadata_table(path):
    """Load a metadata spreadsheet (.csv or .xlsx/.xls) with SampleID, Brand, Grade,
    City, State columns (case-insensitive, any column order; extra columns ignored)."""
    p=Path(path)
    df=pd.read_excel(p) if p.suffix.lower() in (".xlsx",".xls") else pd.read_csv(p)
    colmap={str(c).strip().lower():c for c in df.columns}
    required=["sampleid","brand","grade","city","state"]
    missing=[r for r in required if r not in colmap]
    if missing: raise ValueError(f"Metadata spreadsheet is missing required column(s): {', '.join(missing)}. "
                                  f"Found columns: {', '.join(str(c) for c in df.columns)}")
    return pd.DataFrame({
        "SampleID":df[colmap["sampleid"]].astype(str).str.strip(),
        "Brand":df[colmap["brand"]],"Grade":df[colmap["grade"]],"City":df[colmap["city"]],"State":df[colmap["state"]],
    })

def apply_metadata_table(meta,metadata_df):
    """Merge a metadata spreadsheet into `meta` by exact SampleID match (SampleID is
    expected to be the .spa filename stem). Year is always derived from SampleID.
    Unmatched samples are kept with blank Brand/Grade/City/State and reported."""
    lookup={str(sid).strip():row for sid,row in zip(metadata_df["SampleID"],metadata_df.to_dict("records"))}
    brand=[]; grade=[]; city=[]; state=[]; year=[]; unmatched=[]
    for sid in meta["SampleID"].astype(str):
        row=lookup.get(sid.strip())
        if row is None:
            unmatched.append(sid); brand.append(""); grade.append(""); city.append(""); state.append("")
        else:
            brand.append(row.get("Brand","")); grade.append(row.get("Grade","")); city.append(row.get("City","")); state.append(row.get("State",""))
        year.append(parse_year_from_sample_id(sid))
    meta=meta.copy(); meta["Brand"]=brand; meta["Grade"]=grade; meta["City"]=city; meta["State"]=state; meta["Year"]=year
    return meta,unmatched

def load_spa_files(paths):
    """Load one or more Thermo/Nicolet OMNIC .spa files via SpectroChemPy.

    Returns (meta, wavenumbers, spectra_df) in the same shape as load_ftir_csv
    so both input paths feed the same downstream analysis pipeline. SampleID is
    always the file's stem (name without extension) and rows are kept in the
    same order as `paths`, so metadata spreadsheets can reliably be matched by
    filename. Year is parsed from SampleID (yyyy.nnnn format).
    """
    if scp is None: raise RuntimeError(f"SpectroChemPy is not available: {SCP_IMPORT_ERROR}")
    paths=[Path(p) for p in paths]
    if not paths: raise ValueError("No .spa files were selected.")
    ds=scp.read_omnic(*[str(p) for p in paths], merge=True, sortbydate=False)
    if isinstance(ds,(list,tuple)):
        raise ValueError("Selected .spa files have inconsistent wavenumber axes and cannot be merged. "
                          "Make sure all spectra were collected with the same instrument range/resolution.")
    wn=np.asarray(ds.x.data,dtype=float); data=np.asarray(ds.data,dtype=float)
    if data.ndim==1: data=data[None,:]
    if data.shape[0]!=len(paths):
        raise ValueError(f"Expected {len(paths)} spectra but SpectroChemPy returned {data.shape[0]}; "
                          "cannot reliably match filenames to samples.")
    stems=[p.stem for p in paths]
    meta=pd.DataFrame({"SampleID":stems,"Brand":"","Grade":"","City":"","State":"",
                        "Year":[parse_year_from_sample_id(s) for s in stems]})
    order=np.argsort(wn)
    spectra=pd.DataFrame(data[:,order])
    return meta.reset_index(drop=True),wn[order],spectra.reset_index(drop=True)

def export_spa_csv(paths,csv_path,metadata_file=None):
    """Combine individual .spa files into one nicely formatted wide CSV.

    Produces a plain header row (SampleID, Brand, Grade, City, State, Year,
    then one numeric wavenumber column per point) which load_ftir_csv() already
    supports via its numeric-column-name auto-detection, so the exported file
    can be reloaded directly in "Wide-format CSV" mode, opened in Excel with
    readable column names, or shared/edited like any other dataset.

    If `metadata_file` is given (a spreadsheet with SampleID/Brand/Grade/City/State
    columns), those values are merged in by matching SampleID to each .spa file's
    filename stem; Year is always derived from SampleID (yyyy.nnnn).
    Returns (n_samples, n_wavenumbers, unmatched_sample_ids).
    """
    meta,wn,spectra=load_spa_files(paths)
    unmatched=[]
    if metadata_file:
        metadata_df=load_metadata_table(metadata_file)
        meta,unmatched=apply_metadata_table(meta,metadata_df)
    columns=list(META_DEFAULT)+[f"{w:.4f}" for w in wn]
    out=pd.concat([meta[META_DEFAULT].reset_index(drop=True),spectra.reset_index(drop=True)],axis=1)
    out.columns=columns
    out.to_csv(csv_path,index=False)
    return len(meta),len(wn),unmatched


def load_ftir_csv(path):
    raw=pd.read_csv(path,low_memory=False)
    marker=raw.iloc[0,0] if len(raw) else None
    if isinstance(marker,str) and "wavenumber" in marker.lower():
        start=7; wn=pd.to_numeric(raw.iloc[0,start:],errors="coerce").to_numpy(float)
        meta=raw.iloc[1:,1:start].copy().reset_index(drop=True); meta.columns=META_DEFAULT[:meta.shape[1]]
        spectra=raw.iloc[1:,start:].apply(pd.to_numeric,errors="coerce").reset_index(drop=True)
    else:
        numeric=[]
        for c in raw.columns:
            try: numeric.append((c,float(c)))
            except (TypeError,ValueError): pass
        if not numeric: raise ValueError("No numeric wavenumber columns were detected.")
        cols=[c for c,_ in numeric]; wn=np.array([v for _,v in numeric]); spectra=raw[cols].apply(pd.to_numeric,errors="coerce"); meta=raw.drop(columns=cols)
    good=np.isfinite(wn); wn=wn[good]; spectra=spectra.loc[:,good]
    order=np.argsort(wn)
    return meta.reset_index(drop=True),wn[order],spectra.iloc[:,order].reset_index(drop=True)

def parse_exclusions(text):
    ranges=[]
    for item in text.replace(";",",").split(","):
        item=item.strip()
        if not item: continue
        parts=item.replace("–","-").split("-")
        if len(parts)!=2: raise ValueError(f"Invalid exclusion '{item}'. Use 500-1000, 1800-1900")
        a,b=map(float,parts); ranges.append((min(a,b),max(a,b)))
    return ranges

def to_nddataset(wn,X):
    return NDDataset(X,coordset=[Coord(np.arange(X.shape[0]),title="sample"),Coord(wn,title="wavenumber",units="cm^-1")])

def scp_preprocess(ds,method,window,poly):
    n_points=ds.shape[-1]
    window=max(poly+2,window); window += (window%2==0)
    window=min(window,n_points if n_points%2 else n_points-1)
    if method=="Raw": return ds
    if method=="SNV": return scp.snv(ds)
    if method=="SG smoothing + SNV": return scp.snv(scp.savgol(ds,size=window,order=poly,deriv=0))
    deriv=1 if method=="First derivative + SNV" else 2
    return scp.snv(scp.savgol(ds,size=window,order=poly,deriv=deriv))

def load_dataset(cfg):
    if cfg["mode"]=="spa":
        meta,wn,xdf=load_spa_files(cfg["spa_files"])
        unmatched=[]
        if cfg.get("metadata_file"):
            metadata_df=load_metadata_table(cfg["metadata_file"])
            meta,unmatched=apply_metadata_table(meta,metadata_df)
        return meta,wn,xdf,unmatched
    meta,wn,xdf=load_ftir_csv(cfg["input"]); return meta,wn,xdf,[]

def analyze(cfg,progress=lambda x:None):
    if scp is None: raise RuntimeError(f"SpectroChemPy is not available: {SCP_IMPORT_ERROR}")
    progress(5); meta,wn,xdf,unmatched=load_dataset(cfg)
    mask=(wn>=cfg["min_wn"])&(wn<=cfg["max_wn"])
    for a,b in cfg["exclude"]: mask &= ~((wn>=a)&(wn<=b))
    wn=wn[mask]; xdf=xdf.loc[:,mask]
    if len(wn)<10: raise ValueError("Too few wavenumbers remain after filtering.")
    progress(15); Ximp=SimpleImputer(strategy="median").fit_transform(xdf.to_numpy(float))
    ds=to_nddataset(wn,Ximp)
    progress(25); ds_p=scp_preprocess(ds,cfg["preprocessing"],cfg["window"],cfg["poly"])
    X=np.asarray(ds_p.data,dtype=float)
    n=min(cfg["components"],X.shape[0]-1,X.shape[1])
    progress(35); pca=scp.PCA(n_components=n); pca.fit(ds_p)
    scores=np.asarray(pca.scores.data,dtype=float)
    loadings=np.asarray(pca.loadings.data,dtype=float)
    ev_ratio=np.asarray(pca.explained_variance_ratio.data,dtype=float)/100.0
    cum=np.cumsum(ev_ratio); n95=int(np.searchsorted(cum,.95)+1) if cum[-1]>=.95 else None
    nz=max(2,min(n95 or n,cfg["cluster_pcs"],n)); z=scores[:,:nz]
    progress(45); rows=[]
    for k in range(2,min(cfg["max_k"],len(z)-1)+1):
        lab=KMeans(k,n_init=30,random_state=42).fit_predict(z); rows.append((k,silhouette_score(z,lab)))
    kval=pd.DataFrame(rows,columns=["k","silhouette"]); best_k=int(kval.loc[kval.silhouette.idxmax(),"k"])
    km=KMeans(best_k,n_init=50,random_state=42).fit_predict(z); ag=AgglomerativeClustering(n_clusters=best_k,linkage="ward").fit_predict(z)
    nn=min(5,len(z)-1); d,_=NearestNeighbors(n_neighbors=nn).fit(z).kneighbors(z); eps=float(np.quantile(d[:,-1],cfg["dbscan_quantile"]))
    db=DBSCAN(eps=eps,min_samples=nn).fit_predict(z)
    progress(65); pcs=[f"PC{i+1}" for i in range(n)]
    score_df=pd.concat([meta,pd.DataFrame(scores,columns=pcs)],axis=1); score_df["KMeansCluster"]=km; score_df["AgglomerativeCluster"]=ag; score_df["DBSCANCluster"]=db
    load_df=pd.DataFrame(loadings.T,index=wn,columns=pcs); load_df.index.name="Wavenumber_cm-1"
    summary={"samples":len(meta),"wavenumbers_retained":len(wn),"preprocessing":cfg["preprocessing"],"excluded_ranges":cfg["exclude"],"PC1_percent":100*ev_ratio[0],"PC2_percent":100*ev_ratio[1],"PC1_PC2_percent":100*sum(ev_ratio[:2]),"components_for_95_percent":n95,"variance_in_fitted_PCs_percent":100*cum[-1],"selected_k":best_k,"best_silhouette":float(kval.silhouette.max()),"kmeans_agglomerative_ARI":float(adjusted_rand_score(km,ag)),"dbscan_clusters":len(set(db))-(-1 in db),"dbscan_noise_samples":int((db==-1).sum()),"dbscan_eps":eps,"unmatched_metadata_samples":unmatched}
    out=Path(cfg["output"]); out.mkdir(parents=True,exist_ok=True)
    score_df.to_csv(out/"pca_scores_and_clusters.csv",index=False); load_df.to_csv(out/"pca_loadings.csv"); kval.to_csv(out/"kmeans_silhouette.csv",index=False); (out/"analysis_summary.json").write_text(json.dumps(summary,indent=2))
    progress(80); return {"meta":meta,"wn":wn,"X":X,"ev_ratio":ev_ratio,"loadings_arr":loadings,"scores":scores,"scores_df":score_df,"loadings":load_df,"kval":kval,"summary":summary,"z":z,"unmatched_metadata_samples":unmatched}

class Worker(QObject):
    done=Signal(object); failed=Signal(str); progress=Signal(int)
    def __init__(self,cfg): super().__init__(); self.cfg=cfg
    def run(self):
        try: self.done.emit(analyze(self.cfg,self.progress.emit))
        except Exception: self.failed.emit(traceback.format_exc())

class PlotCanvas(QWidget):
    """Matplotlib canvas with a navigation toolbar (pan/zoom/save) plus
    optional hover tooltips (mplcursors) and click-to-select callbacks,
    giving every plot tab basic interactivity."""
    def __init__(self):
        super().__init__()
        self.fig=Figure(figsize=(8,6),tight_layout=True)
        self.canvas=FigureCanvas(self.fig)
        self.toolbar=NavigationToolbar(self.canvas,self)
        lay=QVBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        lay.addWidget(self.toolbar); lay.addWidget(self.canvas,1)
        self._cursors=[]; self._pick_cid=None
    def clear(self):
        self.fig.clear()
        for cur in self._cursors:
            try: cur.remove()
            except Exception: pass
        self._cursors=[]
        if self._pick_cid is not None:
            try: self.canvas.mpl_disconnect(self._pick_cid)
            except Exception: pass
            self._pick_cid=None
    def draw(self): self.canvas.draw_idle()
    def add_hover(self,artists,formatter):
        """Attach a hover tooltip to one or more artists using mplcursors, if available."""
        if mplcursors is None or not artists: return
        cur=mplcursors.cursor(artists,hover=True)
        @cur.connect("add")
        def _(sel): sel.annotation.set_text(formatter(sel))
        self._cursors.append(cur)
    def on_pick(self,callback):
        self._pick_cid=self.canvas.mpl_connect("pick_event",callback)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("FTIR PCA Workbench (SpectroChemPy)"); self.resize(1400,900); self.result=None; self._build()
        if scp is None:
            QMessageBox.warning(self,"SpectroChemPy not available",
                f"SpectroChemPy could not be imported, so analysis will fail until it is installed:\n{SCP_IMPORT_ERROR}\n\n"
                "Install it with: pip install spectrochempy")
    def _build(self):
        root=QWidget(); self.setCentralWidget(root); layout=QVBoxLayout(root)
        box=QGroupBox("Dataset and model settings"); outer=QVBoxLayout(box)

        mode_row=QHBoxLayout(); mode_row.addWidget(QLabel("Input type:"))
        self.mode_csv=QRadioButton("Wide-format CSV"); self.mode_spa=QRadioButton("Individual .spa files (OMNIC)")
        self.mode_csv.setChecked(True); self.mode_group=QButtonGroup(self); self.mode_group.addButton(self.mode_csv); self.mode_group.addButton(self.mode_spa)
        self.mode_csv.toggled.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_csv); mode_row.addWidget(self.mode_spa); mode_row.addStretch(1); outer.addLayout(mode_row)

        self.input_stack=QStackedWidget(); outer.addWidget(self.input_stack)
        csv_page=QWidget(); csv_row=QHBoxLayout(csv_page); csv_row.setContentsMargins(0,0,0,0)
        self.input=QLineEdit(); bcsv=QPushButton("Browse…"); bcsv.clicked.connect(self.browse_input)
        csv_row.addWidget(QLabel("Input CSV")); csv_row.addWidget(self.input,1); csv_row.addWidget(bcsv)
        self.input_stack.addWidget(csv_page)

        spa_page=QWidget(); spa_col=QVBoxLayout(spa_page); spa_col.setContentsMargins(0,0,0,0)
        self.spa_list=QListWidget(); self.spa_list.setSelectionMode(QAbstractItemView.ExtendedSelection); self.spa_list.setMaximumHeight(90)
        spa_btn_row=QHBoxLayout()
        badd=QPushButton("Add .spa files…"); badd.clicked.connect(self.add_spa_files)
        brem=QPushButton("Remove selected"); brem.clicked.connect(self.remove_spa_files)
        bclr=QPushButton("Clear all"); bclr.clicked.connect(self.clear_spa_files)
        bexp=QPushButton("Export combined CSV…"); bexp.clicked.connect(self.export_spa_csv)
        spa_btn_row.addWidget(badd); spa_btn_row.addWidget(brem); spa_btn_row.addWidget(bclr); spa_btn_row.addWidget(bexp); spa_btn_row.addStretch(1)
        spa_meta_row=QHBoxLayout()
        self.spa_metadata=QLineEdit(); self.spa_metadata.setPlaceholderText("Optional: spreadsheet with SampleID, Brand, Grade, City, State columns")
        bmeta=QPushButton("Load metadata spreadsheet…"); bmeta.clicked.connect(self.browse_spa_metadata)
        bmetaclr=QPushButton("Clear"); bmetaclr.clicked.connect(lambda: self.spa_metadata.setText(""))
        spa_meta_row.addWidget(QLabel("Metadata")); spa_meta_row.addWidget(self.spa_metadata,1); spa_meta_row.addWidget(bmeta); spa_meta_row.addWidget(bmetaclr)
        spa_col.addWidget(self.spa_list); spa_col.addLayout(spa_btn_row); spa_col.addLayout(spa_meta_row)
        self.input_stack.addWidget(spa_page)

        grid=QGridLayout(); outer.addLayout(grid)
        self.output=QLineEdit(str(Path.cwd()/"ftir_pca_results")); bo=QPushButton("Output…"); bo.clicked.connect(self.browse_output)
        self.pre=QComboBox(); self.pre.addItems(["First derivative + SNV","SG smoothing + SNV","SNV","Second derivative + SNV","Raw"])
        self.minwn=self.dspin(400,4000,400); self.maxwn=self.dspin(400,4000,4000); self.exclude=QLineEdit("500-1000")
        self.window=self.spin(5,101,21); self.poly=self.spin(2,5,3); self.components=self.spin(2,100,30); self.clusterpcs=self.spin(2,30,10); self.maxk=self.spin(3,20,10)
        self.dbq=self.dspin(.5,.99,.90,2,.01)
        controls=[("Output folder",self.output,bo),("Preprocessing",self.pre,None),("Analysis range minimum",self.minwn,None),("Analysis range maximum",self.maxwn,None),("Excluded ranges",self.exclude,None),("SG window",self.window,None),("SG polynomial",self.poly,None),("PCA components",self.components,None),("PCs used for clustering",self.clusterpcs,None),("Maximum K",self.maxk,None),("DBSCAN neighbor quantile",self.dbq,None)]
        for i,(lab,w,extra) in enumerate(controls): r=i//3; c=(i%3)*3; grid.addWidget(QLabel(lab),r,c); grid.addWidget(w,r,c+1); extra and grid.addWidget(extra,r,c+2)
        layout.addWidget(box)
        row=QHBoxLayout(); self.runbtn=QPushButton("Run analysis"); self.runbtn.clicked.connect(self.run); self.runbtn.setMinimumHeight(38); self.progress=QProgressBar(); row.addWidget(self.runbtn); row.addWidget(self.progress); layout.addLayout(row)
        self.tabs=QTabWidget(); layout.addWidget(self.tabs,1)
        self.summary=QPlainTextEdit(); self.summary.setReadOnly(True); self.tabs.addTab(self.summary,"Summary")
        self.canvases={}
        for key,title in [("spectra","Spectra"),("scores","PCA Scores"),("loadings","Loadings"),("variance","Variance"),("clusters","Clusters"),("dendro","Dendrogram")]: c=PlotCanvas(); self.canvases[key]=c; self.tabs.addTab(c,title)
        self.table=QTableWidget(); self.tabs.addTab(self.table,"Scores table")
    def _on_mode_changed(self,checked):
        self.input_stack.setCurrentIndex(0 if self.mode_csv.isChecked() else 1)
    def spin(self,a,b,v): w=QSpinBox(); w.setRange(a,b); w.setValue(v); return w
    def dspin(self,a,b,v,dec=1,step=1): w=QDoubleSpinBox(); w.setRange(a,b); w.setDecimals(dec); w.setSingleStep(step); w.setValue(v); return w
    def browse_input(self):
        p,_=QFileDialog.getOpenFileName(self,"Select FTIR CSV","","CSV files (*.csv);;All files (*)")
        if p: self.input.setText(p); self.output.setText(str(Path(p).with_name(Path(p).stem+"_pca_results")))
    def add_spa_files(self):
        ps,_=QFileDialog.getOpenFileNames(self,"Select individual FTIR .spa files","","OMNIC spectra (*.spa);;All files (*)")
        if not ps: return
        existing={self.spa_list.item(i).data(Qt.UserRole) for i in range(self.spa_list.count())}
        for p in ps:
            if p not in existing:
                it=QListWidgetItem(Path(p).name); it.setData(Qt.UserRole,p); self.spa_list.addItem(it)
        if self.spa_list.count() and not self.output.text().strip():
            first=self.spa_list.item(0).data(Qt.UserRole)
            self.output.setText(str(Path(first).with_name(Path(first).stem+"_pca_results")))
    def remove_spa_files(self):
        for it in self.spa_list.selectedItems(): self.spa_list.takeItem(self.spa_list.row(it))
    def clear_spa_files(self): self.spa_list.clear()
    def browse_spa_metadata(self):
        p,_=QFileDialog.getOpenFileName(self,"Select metadata spreadsheet","","Spreadsheets (*.csv *.xlsx *.xls);;All files (*)")
        if p: self.spa_metadata.setText(p)
    def _warn_unmatched(self,unmatched):
        if not unmatched: return
        shown=", ".join(unmatched[:20])+(", …" if len(unmatched)>20 else "")
        QMessageBox.warning(self,"Unmatched samples",
            f"{len(unmatched)} .spa sample(s) had no matching SampleID in the metadata spreadsheet "
            f"and were kept with blank Brand/Grade/City/State:\n{shown}")
    def export_spa_csv(self):
        files=[self.spa_list.item(i).data(Qt.UserRole) for i in range(self.spa_list.count())]
        if not files: QMessageBox.warning(self,"Export combined CSV","Add at least one .spa file first."); return
        default=str(Path(files[0]).with_name("combined_ftir_spectra.csv"))
        p,_=QFileDialog.getSaveFileName(self,"Save combined CSV",default,"CSV files (*.csv)")
        if not p: return
        try:
            n_samples,n_wn,unmatched=export_spa_csv(files,p,self.spa_metadata.text().strip() or None)
        except Exception as e:
            QMessageBox.critical(self,"Export failed",str(e)); return
        QMessageBox.information(self,"Export combined CSV",f"Wrote {n_samples} samples x {n_wn} wavenumbers to:\n{p}")
        self._warn_unmatched(unmatched)
    def browse_output(self):
        p=QFileDialog.getExistingDirectory(self,"Select output folder")
        if p: self.output.setText(p)
    def config(self):
        mode="spa" if self.mode_spa.isChecked() else "csv"
        cfg={"mode":mode,"input":self.input.text().strip(),
             "spa_files":[self.spa_list.item(i).data(Qt.UserRole) for i in range(self.spa_list.count())],
             "metadata_file":self.spa_metadata.text().strip() or None,
             "output":self.output.text().strip(),"preprocessing":self.pre.currentText(),"min_wn":self.minwn.value(),"max_wn":self.maxwn.value(),
             "exclude":parse_exclusions(self.exclude.text()),"window":self.window.value(),"poly":self.poly.value(),"components":self.components.value(),
             "cluster_pcs":self.clusterpcs.value(),"max_k":self.maxk.value(),"dbscan_quantile":self.dbq.value()}
        return cfg
    def run(self):
        try:
            cfg=self.config()
            if cfg["mode"]=="csv" and not Path(cfg["input"]).exists(): raise ValueError("Select an input CSV first.")
            if cfg["mode"]=="spa" and not cfg["spa_files"]: raise ValueError("Add at least one .spa file first.")
            if not cfg["output"]: raise ValueError("Choose an output folder.")
        except Exception as e: QMessageBox.warning(self,"Settings",str(e)); return
        self.runbtn.setEnabled(False); self.progress.setValue(0); self.summary.setPlainText("Running analysis…")
        self.thread=QThread(); self.worker=Worker(cfg); self.worker.moveToThread(self.thread); self.thread.started.connect(self.worker.run); self.worker.progress.connect(self.progress.setValue); self.worker.done.connect(self.finished); self.worker.failed.connect(self.failed); self.worker.done.connect(self.thread.quit); self.worker.failed.connect(self.thread.quit); self.thread.start()
    def failed(self,text): self.runbtn.setEnabled(True); self.summary.setPlainText(text); QMessageBox.critical(self,"Analysis failed",text.splitlines()[-1])
    def finished(self,r):
        self.result=r; self.runbtn.setEnabled(True); self.progress.setValue(100); self.summary.setPlainText(json.dumps(r["summary"],indent=2)); self.draw(); self.populate_table(r["scores_df"])
        self._warn_unmatched(r.get("unmatched_metadata_samples") or [])
    def _select_table_row(self,idx):
        if 0<=idx<self.table.rowCount(): self.table.selectRow(idx); self.table.scrollToItem(self.table.item(idx,0))
    def draw(self):
        r=self.result; wn=r["wn"]; X=r["X"]; ev=r["ev_ratio"]; loadings=r["loadings_arr"]; df=r["scores_df"].reset_index(drop=True)
        sample_labels=df.get("SampleID",pd.Series(range(len(df)))).astype(str).tolist()

        c=self.canvases["spectra"]; c.clear(); ax=c.fig.add_subplot()
        nshow=min(80,len(X)); lines=ax.plot(wn,X[:nshow].T,alpha=.35,lw=.7,picker=3)
        ax.invert_xaxis(); ax.set(title="Preprocessed spectra (hover/click a line to identify)",xlabel="Wavenumber (cm⁻¹)",ylabel="Processed intensity")
        c.add_hover(lines,lambda sel: sample_labels[lines.index(sel.artist)] if sel.artist in lines else "")
        c.draw()

        c=self.canvases["scores"]; c.clear(); ax=c.fig.add_subplot(); score_artists=[]; score_idx=[]
        for lab,g in df.groupby("Brand",dropna=False):
            sc=ax.scatter(g.PC1,g.PC2,s=35,alpha=.75,label=str(lab),picker=5); score_artists.append(sc); score_idx.append(g.index.to_numpy())
        ax.set(xlabel=f"PC1 ({ev[0]*100:.1f}%)",ylabel=f"PC2 ({ev[1]*100:.1f}%)",title="PCA scores by brand (hover/click a point to identify)"); ax.legend(fontsize=7,ncol=2)
        c.add_hover(score_artists,lambda sel: sample_labels[score_idx[score_artists.index(sel.artist)][sel.index]])
        def _pick_scores(event):
            if event.artist in score_artists:
                gi=score_idx[score_artists.index(event.artist)]; self._select_table_row(int(gi[event.ind[0]]))
        c.on_pick(_pick_scores); c.draw()

        c=self.canvases["loadings"]; c.clear(); axs=c.fig.subplots(3,1,sharex=True)
        for i,ax in enumerate(axs): ax.plot(wn,loadings[i],lw=.8); ax.set_ylabel(f"PC{i+1}")
        axs[-1].invert_xaxis(); axs[-1].set_xlabel("Wavenumber (cm⁻¹)"); c.draw()

        c=self.canvases["variance"]; c.clear(); ax=c.fig.add_subplot(); ax.bar(range(1,len(ev)+1),ev*100,picker=5); ax.plot(range(1,len(ev)+1),np.cumsum(ev)*100,color="darkred",marker="o",ms=3); ax.axhline(95,ls="--",color="gray"); ax.set(title="Explained variance",xlabel="Principal component",ylabel="Percent"); c.draw()

        c=self.canvases["clusters"]; c.clear(); ax=c.fig.add_subplot(); clus_artists=[]; clus_idx=[]
        for lab,g in df.groupby("KMeansCluster"):
            sc=ax.scatter(g.PC1,g.PC2,s=45,alpha=.8,label=f"Cluster {lab}",picker=5); clus_artists.append(sc); clus_idx.append(g.index.to_numpy())
        ax.legend(); ax.set(title="K-Means clusters on PCA scores (hover/click a point to identify)",xlabel="PC1",ylabel="PC2")
        c.add_hover(clus_artists,lambda sel: sample_labels[clus_idx[clus_artists.index(sel.artist)][sel.index]])
        def _pick_clusters(event):
            if event.artist in clus_artists:
                gi=clus_idx[clus_artists.index(event.artist)]; self._select_table_row(int(gi[event.ind[0]]))
        c.on_pick(_pick_clusters); c.draw()

        c=self.canvases["dendro"]; c.clear(); ax=c.fig.add_subplot(); dendrogram(linkage(r["z"],method="ward"),labels=sample_labels,leaf_rotation=90,leaf_font_size=4,ax=ax); ax.set(title="Ward hierarchical clustering",ylabel="Distance"); c.draw()
    def populate_table(self,df):
        show=[c for c in ["SampleID","Brand","Grade","City","State","PC1","PC2","PC3","KMeansCluster","AgglomerativeCluster","DBSCANCluster"] if c in df]
        self.table.setRowCount(len(df)); self.table.setColumnCount(len(show)); self.table.setHorizontalHeaderLabels(show)
        for i,row in df[show].reset_index(drop=True).iterrows():
            for j,v in enumerate(row): self.table.setItem(i,j,QTableWidgetItem(f"{v:.5g}" if isinstance(v,(float,np.floating)) else str(v)))
        self.table.resizeColumnsToContents()

if __name__=="__main__":
    app=QApplication(sys.argv); app.setStyle("Fusion"); w=MainWindow()
    args=sys.argv[1:]
    if args:
        if all(Path(a).suffix.lower()==".spa" for a in args):
            w.mode_spa.setChecked(True)
            for a in args:
                it=QListWidgetItem(Path(a).name); it.setData(Qt.UserRole,a); w.spa_list.addItem(it)
            w.output.setText(str(Path(args[0]).with_name(Path(args[0]).stem+"_pca_results")))
        else:
            w.mode_csv.setChecked(True); w.input.setText(args[0]); w.output.setText(str(Path(args[0]).with_name(Path(args[0]).stem+"_pca_results")))
    w.show(); sys.exit(app.exec())
