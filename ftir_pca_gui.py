#!/usr/bin/env python3
"""PySide6 desktop UI for reusable FTIR PCA and clustering analysis."""
from __future__ import annotations
import json, sys, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QGridLayout,
 QGroupBox,QLabel,QLineEdit,QPushButton,QFileDialog,QComboBox,QSpinBox,QDoubleSpinBox,QCheckBox,
 QTabWidget,QPlainTextEdit,QTableWidget,QTableWidgetItem,QMessageBox,QProgressBar,QSplitter)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

META_DEFAULT=["SampleID","Brand","Grade","City","State","Year"]

def load_ftir(path):
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

def snv(x):
    sd=x.std(axis=1,keepdims=True); sd[sd==0]=1
    return (x-x.mean(axis=1,keepdims=True))/sd

def preprocess(x,method,window,poly):
    x=SimpleImputer(strategy="median").fit_transform(x)
    window=max(poly+2,window); window += (window%2==0)
    window=min(window,x.shape[1] if x.shape[1]%2 else x.shape[1]-1)
    if method=="Raw": return x
    if method=="SNV": return snv(x)
    if method=="SG smoothing + SNV": return snv(savgol_filter(x,window,poly,axis=1))
    deriv=1 if method=="First derivative + SNV" else 2
    return snv(savgol_filter(x,window,poly,deriv=deriv,axis=1))

def analyze(cfg,progress=lambda x:None):
    progress(5); meta,wn,xdf=load_ftir(cfg["input"])
    mask=(wn>=cfg["min_wn"])&(wn<=cfg["max_wn"])
    for a,b in cfg["exclude"]: mask &= ~((wn>=a)&(wn<=b))
    wn=wn[mask]; xdf=xdf.loc[:,mask]
    if len(wn)<10: raise ValueError("Too few wavenumbers remain after filtering.")
    progress(20); X=preprocess(xdf.to_numpy(float),cfg["preprocessing"],cfg["window"],cfg["poly"]); X-=X.mean(axis=0)
    n=min(cfg["components"],len(X)-1,X.shape[1]); pca=PCA(n_components=n,svd_solver="full").fit(X); scores=pca.transform(X)
    cum=np.cumsum(pca.explained_variance_ratio_); n95=int(np.searchsorted(cum,.95)+1) if cum[-1]>=.95 else None
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
    load_df=pd.DataFrame(pca.components_.T,index=wn,columns=pcs); load_df.index.name="Wavenumber_cm-1"
    summary={"samples":len(meta),"wavenumbers_retained":len(wn),"preprocessing":cfg["preprocessing"],"excluded_ranges":cfg["exclude"],"PC1_percent":100*pca.explained_variance_ratio_[0],"PC2_percent":100*pca.explained_variance_ratio_[1],"PC1_PC2_percent":100*sum(pca.explained_variance_ratio_[:2]),"components_for_95_percent":n95,"variance_in_fitted_PCs_percent":100*cum[-1],"selected_k":best_k,"best_silhouette":float(kval.silhouette.max()),"kmeans_agglomerative_ARI":float(adjusted_rand_score(km,ag)),"dbscan_clusters":len(set(db))-(-1 in db),"dbscan_noise_samples":int((db==-1).sum()),"dbscan_eps":eps}
    out=Path(cfg["output"]); out.mkdir(parents=True,exist_ok=True)
    score_df.to_csv(out/"pca_scores_and_clusters.csv",index=False); load_df.to_csv(out/"pca_loadings.csv"); kval.to_csv(out/"kmeans_silhouette.csv",index=False); (out/"analysis_summary.json").write_text(json.dumps(summary,indent=2))
    progress(80); return {"meta":meta,"wn":wn,"X":X,"pca":pca,"scores":scores,"scores_df":score_df,"loadings":load_df,"kval":kval,"summary":summary,"z":z}

class Worker(QObject):
    done=Signal(object); failed=Signal(str); progress=Signal(int)
    def __init__(self,cfg): super().__init__(); self.cfg=cfg
    def run(self):
        try: self.done.emit(analyze(self.cfg,self.progress.emit))
        except Exception: self.failed.emit(traceback.format_exc())

class PlotCanvas(FigureCanvas):
    def __init__(self): self.fig=Figure(figsize=(8,6),tight_layout=True); super().__init__(self.fig)
    def clear(self): self.fig.clear()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("FTIR PCA Workbench"); self.resize(1400,900); self.result=None; self._build()
    def _build(self):
        root=QWidget(); self.setCentralWidget(root); layout=QVBoxLayout(root)
        box=QGroupBox("Dataset and model settings"); grid=QGridLayout(box)
        self.input=QLineEdit(); b=QPushButton("Browse…"); b.clicked.connect(self.browse_input)
        self.output=QLineEdit(str(Path.cwd()/"ftir_pca_results")); bo=QPushButton("Output…"); bo.clicked.connect(self.browse_output)
        self.pre=QComboBox(); self.pre.addItems(["First derivative + SNV","SG smoothing + SNV","SNV","Second derivative + SNV","Raw"])
        self.minwn=self.dspin(400,4000,400); self.maxwn=self.dspin(400,4000,4000); self.exclude=QLineEdit("500-1000")
        self.window=self.spin(5,101,21); self.poly=self.spin(2,5,3); self.components=self.spin(2,100,30); self.clusterpcs=self.spin(2,30,10); self.maxk=self.spin(3,20,10)
        self.dbq=self.dspin(.5,.99,.90,2,.01)
        controls=[("Input CSV",self.input,b),("Output folder",self.output,bo),("Preprocessing",self.pre,None),("Analysis range minimum",self.minwn,None),("Analysis range maximum",self.maxwn,None),("Excluded ranges",self.exclude,None),("SG window",self.window,None),("SG polynomial",self.poly,None),("PCA components",self.components,None),("PCs used for clustering",self.clusterpcs,None),("Maximum K",self.maxk,None),("DBSCAN neighbor quantile",self.dbq,None)]
        for i,(lab,w,extra) in enumerate(controls): r=i//3; c=(i%3)*3; grid.addWidget(QLabel(lab),r,c); grid.addWidget(w,r,c+1); extra and grid.addWidget(extra,r,c+2)
        layout.addWidget(box)
        row=QHBoxLayout(); self.runbtn=QPushButton("Run analysis"); self.runbtn.clicked.connect(self.run); self.runbtn.setMinimumHeight(38); self.progress=QProgressBar(); row.addWidget(self.runbtn); row.addWidget(self.progress); layout.addLayout(row)
        self.tabs=QTabWidget(); layout.addWidget(self.tabs,1)
        self.summary=QPlainTextEdit(); self.summary.setReadOnly(True); self.tabs.addTab(self.summary,"Summary")
        self.canvases={}
        for key,title in [("spectra","Spectra"),("scores","PCA Scores"),("loadings","Loadings"),("variance","Variance"),("clusters","Clusters"),("dendro","Dendrogram")]: c=PlotCanvas(); self.canvases[key]=c; self.tabs.addTab(c,title)
        self.table=QTableWidget(); self.tabs.addTab(self.table,"Scores table")
    def spin(self,a,b,v): w=QSpinBox(); w.setRange(a,b); w.setValue(v); return w
    def dspin(self,a,b,v,dec=1,step=1): w=QDoubleSpinBox(); w.setRange(a,b); w.setDecimals(dec); w.setSingleStep(step); w.setValue(v); return w
    def browse_input(self):
        p,_=QFileDialog.getOpenFileName(self,"Select FTIR CSV","","CSV files (*.csv);;All files (*)")
        if p: self.input.setText(p); self.output.setText(str(Path(p).with_name(Path(p).stem+"_pca_results")))
    def browse_output(self):
        p=QFileDialog.getExistingDirectory(self,"Select output folder")
        if p: self.output.setText(p)
    def config(self): return {"input":self.input.text().strip(),"output":self.output.text().strip(),"preprocessing":self.pre.currentText(),"min_wn":self.minwn.value(),"max_wn":self.maxwn.value(),"exclude":parse_exclusions(self.exclude.text()),"window":self.window.value(),"poly":self.poly.value(),"components":self.components.value(),"cluster_pcs":self.clusterpcs.value(),"max_k":self.maxk.value(),"dbscan_quantile":self.dbq.value()}
    def run(self):
        try:
            cfg=self.config()
            if not Path(cfg["input"]).exists(): raise ValueError("Select an input CSV first.")
        except Exception as e: QMessageBox.warning(self,"Settings",str(e)); return
        self.runbtn.setEnabled(False); self.progress.setValue(0); self.summary.setPlainText("Running analysis…")
        self.thread=QThread(); self.worker=Worker(cfg); self.worker.moveToThread(self.thread); self.thread.started.connect(self.worker.run); self.worker.progress.connect(self.progress.setValue); self.worker.done.connect(self.finished); self.worker.failed.connect(self.failed); self.worker.done.connect(self.thread.quit); self.worker.failed.connect(self.thread.quit); self.thread.start()
    def failed(self,text): self.runbtn.setEnabled(True); self.summary.setPlainText(text); QMessageBox.critical(self,"Analysis failed",text.splitlines()[-1])
    def finished(self,r):
        self.result=r; self.runbtn.setEnabled(True); self.progress.setValue(100); self.summary.setPlainText(json.dumps(r["summary"],indent=2)); self.draw(); self.populate_table(r["scores_df"])
    def draw(self):
        r=self.result; wn=r["wn"]; X=r["X"]; pca=r["pca"]; s=r["scores"]; df=r["scores_df"]
        c=self.canvases["spectra"]; c.clear(); ax=c.fig.add_subplot(); ax.plot(wn,X[:min(80,len(X))].T,alpha=.16,lw=.6); ax.invert_xaxis(); ax.set(title="Preprocessed spectra",xlabel="Wavenumber (cm⁻¹)",ylabel="Processed intensity"); c.draw()
        c=self.canvases["scores"]; c.clear(); ax=c.fig.add_subplot();
        for lab,g in df.groupby("Brand",dropna=False): ax.scatter(g.PC1,g.PC2,s=35,alpha=.75,label=str(lab))
        ax.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",ylabel=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",title="PCA scores by brand"); ax.legend(fontsize=7,ncol=2); c.draw()
        c=self.canvases["loadings"]; c.clear(); axs=c.fig.subplots(3,1,sharex=True)
        for i,ax in enumerate(axs): ax.plot(wn,pca.components_[i],lw=.8); ax.set_ylabel(f"PC{i+1}")
        axs[-1].invert_xaxis(); axs[-1].set_xlabel("Wavenumber (cm⁻¹)"); c.draw()
        c=self.canvases["variance"]; c.clear(); ax=c.fig.add_subplot(); ev=pca.explained_variance_ratio_; ax.bar(range(1,len(ev)+1),ev*100); ax.plot(range(1,len(ev)+1),np.cumsum(ev)*100,color="darkred",marker="o",ms=3); ax.axhline(95,ls="--",color="gray"); ax.set(title="Explained variance",xlabel="Principal component",ylabel="Percent"); c.draw()
        c=self.canvases["clusters"]; c.clear(); ax=c.fig.add_subplot();
        for lab,g in df.groupby("KMeansCluster"): ax.scatter(g.PC1,g.PC2,s=45,alpha=.8,label=f"Cluster {lab}")
        ax.legend(); ax.set(title="K-Means clusters on PCA scores",xlabel="PC1",ylabel="PC2"); c.draw()
        c=self.canvases["dendro"]; c.clear(); ax=c.fig.add_subplot(); labels=df.get("SampleID",pd.Series(range(len(df)))).astype(str).tolist(); dendrogram(linkage(r["z"],method="ward"),labels=labels,leaf_rotation=90,leaf_font_size=4,ax=ax); ax.set(title="Ward hierarchical clustering",ylabel="Distance"); c.draw()
    def populate_table(self,df):
        show=[c for c in ["SampleID","Brand","Grade","City","State","PC1","PC2","PC3","KMeansCluster","AgglomerativeCluster","DBSCANCluster"] if c in df]
        self.table.setRowCount(len(df)); self.table.setColumnCount(len(show)); self.table.setHorizontalHeaderLabels(show)
        for i,row in df[show].iterrows():
            for j,v in enumerate(row): self.table.setItem(i,j,QTableWidgetItem(f"{v:.5g}" if isinstance(v,(float,np.floating)) else str(v)))
        self.table.resizeColumnsToContents()

if __name__=="__main__":
    app=QApplication(sys.argv); app.setStyle("Fusion"); w=MainWindow()
    if len(sys.argv)>1: w.input.setText(sys.argv[1]); w.output.setText(str(Path(sys.argv[1]).with_name(Path(sys.argv[1]).stem+"_pca_results")))
    w.show(); sys.exit(app.exec())
