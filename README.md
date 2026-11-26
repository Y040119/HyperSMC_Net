# HyperSMC-Net

**HyperSMC-Net** is an epigenetically constrained spatial multi-omics framework for inferring cell–cell communication (CCC) and downstream intracellular gene regulatory networks (GRNs) across spatial transcriptomics and chromatin accessibility data.

## Repository Structure

```text
HyperSMC_Net/
├── preprocess_epi.py   # Dual-omics preprocessing, promoter gene activity scoring, and epigenetically constrained spatial interface construction
├── ser_model.py        # Spatial Epigenetic Regulon (SER) inference and Regulon Activity Score (RAS) quantification
├── hgnn_sigxtalk.py    # Dynamic hypergraph neural network (HGNNP) for L–R–TF–Target cascade scoring and candidate discovery
├── ot.py               # Sparse unbalanced/combined optimal transport (COT) solvers for spatial ligand–receptor coupling
├── LICENSE             # MIT License
└── README.md           # Documentation and usage guide
```

## Installation & Dependencies

HyperSMC-Net is implemented in Python ($\ge 3.8$). Core dependencies include:

* `numpy`, `pandas`, `scipy`, `scikit-learn`, `tqdm`, `joblib`
* `scanpy`, `anndata`, `magic-impute`, `scprep`
* `libpysal`, `esda`
* `torch` (PyTorch)
* `dhg` (DeepHypergraph)

You can install the required packages via `pip`:

```bash
pip install numpy pandas scipy scikit-learn tqdm joblib scanpy anndata magic-impute scprep libpysal esda torch dhg
```

## Prior Databases Setup

Place species-specific prior knowledge tables (`lrpairs.tsv`, `pathway.tsv`, and TF–target motif databases) under a local `database/` directory (or specify a custom root via the `HYPERSMC_DB_ROOT` environment variable):

```text
database/
├── human/
│   ├── lrpairs.tsv
│   ├── pathway.tsv
│   └── Human_TF_Target.tsv
└── mouse/
    ├── lrpairs.tsv
    ├── pathway.tsv
    └── Mouse_TF_Target.tsv
```

## Quick Start

Below is a minimal end-to-end workflow demonstrating how the four core modules connect:

```python
import torch
import preprocess_epi as pp_epi
from ser_model import SERInferrer
import hgnn_sigxtalk as hgnn

# 1. Dual-omics preprocessing & gene activity construction
# adata_rna and adata_atac are AnnData objects sharing .obsm['spatial']
adata_activity = pp_epi.build_gene_activity_matrix(adata_atac, gtf_path="gencode.vM25.annotation.gtf")
adata_rna, adata_activity = pp_epi.find_dual_svgs(adata_rna, adata_activity)

# 2. Epigenetically constrained spatial cell interface construction
lr_df = pp_epi.load_lr_df(is_human=False)
valid_genes = set(adata_rna.var_names)
lr_df = lr_df[lr_df['ligand'].isin(valid_genes) & lr_df['receptor'].isin(valid_genes)].copy()
lr_df.index = lr_df['ligand'] + "_" + lr_df['receptor']

pairs, pairs_meta = pp_epi.find_interfaces(
    adata_rna,
    cluster_key='type',
    lr_df=lr_df,
    adata_activity=adata_activity,
    alpha=0.5
)

pathway_df = pp_epi.load_pathway_df(is_human=False)
pathway_df = pathway_df[pathway_df['src'].isin(valid_genes) & pathway_df['dest'].isin(valid_genes)]
lr_raw = pp_epi.subset_lr(is_human=False)

score_mat, direction_mat = pp_epi.score(adata_rna, lr_df, pairs, pairs_meta)
idata = pp_epi.idata_construct(score_mat, direction_mat, pairs_meta, lr_df, lr_raw, pathway_df, adata_rna)

# 3. Spatial Epigenetic Regulon (SER) inference
ser = SERInferrer(adata_rna, adata_activity, n_neighbors=8)
tf_targets = ser.load_motif_database("database/mouse/Mouse_TF_Target.tsv", source='custom')
adata_ras, regulons = ser.infer(
    tf_targets,
    global_corr_th=0.05,
    spatial_corr_th=0.05,
    min_activity_th=0.0001,
    min_targets=1,
    moran_I=0.04
)

# 4. Dynamic hypergraph scoring of L-R-TF-Target cascades
device = "cuda" if torch.cuda.is_available() else "cpu"
cascades_df = hgnn.run_sigxtalk_scoring(
    adata_rna=adata_rna,
    idata=idata,
    ras_df=adata_ras.to_df(),
    regulons=regulons,
    device=device,
    epochs=1000,
    pathway_df=pathway_df,
    min_corr=0.06,
    min_co_rate=0.15,
    min_rtgt_corr=0.05
)

print(cascades_df.head())
```

## Hyperparameter Configurations

Dataset-specific hyperparameter settings used across all benchmark and biological case studies (MISAR-seq E11.0–E18.5, P22 mouse brain epigenomic modalities, human hippocampus, and simulated datasets) are documented in **Additional file 1 (Table S1)** of the manuscript.

## License

This project is released under the [MIT License](LICENSE).
