import warnings
import anndata
from esda.moran import Moran
from joblib import Parallel, delayed
import libpysal
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats
from tqdm import tqdm


class SERInferrer:
    def __init__(self, adata_rna, adata_activity, n_neighbors=15):
        self.adata_rna = adata_rna.copy()
        self.adata_atac = adata_activity.copy()

        common_cells = np.intersect1d(self.adata_rna.obs_names, self.adata_atac.obs_names)
        if len(common_cells) < len(self.adata_rna.obs_names):
            self.adata_rna = self.adata_rna[common_cells, :].copy()
            self.adata_atac = self.adata_atac[common_cells, :].copy()

        if 'connectivities' not in self.adata_rna.obsp:
            sc.pp.neighbors(self.adata_rna, use_rep='spatial', n_neighbors=n_neighbors)

        self.connectivities = self.adata_rna.obsp['connectivities']

    def load_motif_database(self, motif_file, source='custom'):
        tf_targets = {}
        try:
            if source == 'spider_pathway':
                df = pd.read_csv(motif_file, sep='	', index_col=0)
                pairs = df[['src', 'dest']].values
            else:
                df = pd.read_csv(motif_file, sep=None, engine='python')
                pairs = df.iloc[:, 0:2].values

            valid_genes_rna = set(self.adata_rna.var_names)

            for tf, target in pairs:
                if tf in valid_genes_rna:
                    if tf not in tf_targets:
                        tf_targets[tf] = set()
                    tf_targets[tf].add(target)

            return tf_targets

        except Exception as e:
            warnings.warn(f"Error loading motif DB: {e}")
            return {}

    def infer(
        self,
        tf_targets_dict,
        global_corr_th=0.3,
        spatial_corr_th=0.2,
        min_activity_th=0.01,
        min_targets=3,
        moran_I=0,
        n_jobs=16
    ):
        if sparse.issparse(self.adata_rna.X):
            X_raw = self.adata_rna.X
        else:
            X_raw = sparse.csr_matrix(self.adata_rna.X)

        w_pysal = libpysal.weights.WSP(self.connectivities).to_W()
        w_pysal.transform = 'r'
        if 'col' in self.adata_rna.obs and 'row' in self.adata_rna.obs:
            coords = self.adata_rna.obs[['col', 'row']].values
        elif 'spatial' in self.adata_rna.obsm:
            coords = self.adata_rna.obsm['spatial']
        else:
            raise ValueError("Spatial coordinates not found in adata_rna.")

        w_moran = libpysal.weights.KNN.from_array(coords, k=6)
        w_moran.transform = 'r'

        W_sparse = w_pysal.sparse
        X_smooth = W_sparse.dot(X_raw)

        if sparse.issparse(self.adata_atac.X):
            X_atac = self.adata_atac.X
        else:
            X_atac = sparse.csr_matrix(self.adata_atac.X)

        gene_list_rna = self.adata_rna.var_names.tolist()
        gene_to_idx_rna = {g: i for i, g in enumerate(gene_list_rna)}

        gene_list_atac = self.adata_atac.var_names.tolist()
        gene_to_idx_atac = {g: i for i, g in enumerate(gene_list_atac)}

        final_regulons = {}
        ras_dict = {}

        def _process_single_tf(tf, targets):
            if tf not in gene_to_idx_rna:
                return tf, None, None

            tf_idx = gene_to_idx_rna[tf]
            tf_expr = X_raw[:, tf_idx].toarray().flatten().astype(np.float64)
            tf_smooth = X_smooth[:, tf_idx].toarray().flatten().astype(np.float64)

            if np.count_nonzero(tf_expr) < 10:
                return tf, None, None

            mi_global = Moran(tf_expr, w_moran, permutations=999)
            passed_global = (mi_global.I > moran_I) and (mi_global.p_sim <= 0.05)

            if not passed_global:
                return tf, None, None

            target_rna_indices = [gene_to_idx_rna[t] for t in targets if t in gene_to_idx_rna]
            if len(target_rna_indices) < min_targets:
                return tf, None, None

            candidate_targets = {}

            for target in targets:
                if target not in gene_to_idx_rna or target not in gene_to_idx_atac:
                    continue

                target_idx_rna = gene_to_idx_rna[target]
                target_idx_atac = gene_to_idx_atac[target]

                target_atac_vec = X_atac[:, target_idx_atac].toarray().flatten()
                atac_mean = target_atac_vec.mean()
                if atac_mean < 1e-6:
                    continue

                target_expr = X_raw[:, target_idx_rna].toarray().flatten()
                g_corr, _ = stats.spearmanr(tf_expr, target_expr)
                if np.isnan(g_corr):
                    g_corr = 0.0

                target_smooth = X_smooth[:, target_idx_rna].toarray().flatten()
                s_corr, _ = stats.spearmanr(tf_smooth, target_smooth)
                if np.isnan(s_corr):
                    s_corr = 0.0

                if s_corr < spatial_corr_th and g_corr < global_corr_th:
                    continue

                pseudo_count = 1e-6
                atac_conf = min(1.0, np.sqrt(atac_mean + pseudo_count) / np.sqrt(min_activity_th + pseudo_count))
                composite_corr = s_corr + 0.2 * max(0.0, g_corr)
                regulatory_score = atac_conf * composite_corr

                if regulatory_score > 0:
                    candidate_targets[target] = regulatory_score

            valid_targets = []
            if len(candidate_targets) > 0:
                dynamic_th = 0.02
                for tgt, score in candidate_targets.items():
                    if score >= dynamic_th:
                        valid_targets.append(tgt)

            if len(valid_targets) >= min_targets:
                ras_vector = self._calculate_ras(
                    valid_targets,
                    gene_to_idx_rna,
                    gene_to_idx_atac,
                    X_raw,
                    X_atac
                )
                return tf, valid_targets, ras_vector

            return tf, None, None

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_process_single_tf)(tf, targets)
            for tf, targets in tqdm(tf_targets_dict.items(), desc="Pruning Regulons")
        )

        for tf, valid_targets, ras_vector in results:
            if valid_targets is not None and ras_vector is not None:
                final_regulons[tf] = valid_targets
                ras_dict[tf] = ras_vector

        if len(ras_dict) > 0:
            ras_df = pd.DataFrame(ras_dict, index=self.adata_rna.obs_names)
            adata_ras = anndata.AnnData(X=ras_df.values, obs=self.adata_rna.obs)
            adata_ras.var_names = ras_df.columns
            adata_ras.obsm['spatial'] = self.adata_rna.obsm['spatial']
            return adata_ras, final_regulons

        warnings.warn("No valid regulons found with current thresholds.")
        return None, {}

    def _calculate_ras(self, targets, gene_to_idx_rna, gene_to_idx_atac, X_expr, X_atac):
        ras_vector = np.zeros(X_expr.shape[0])

        for target in targets:
            idx_rna = gene_to_idx_rna[target]
            idx_atac = gene_to_idx_atac[target]

            t_expr = X_expr[:, idx_rna].toarray().flatten()
            t_act = X_atac[:, idx_atac].toarray().flatten()

            q99_expr = np.quantile(t_expr, 0.99)
            if q99_expr > 1e-6:
                t_expr = t_expr / q99_expr
                t_expr[t_expr > 1.0] = 1.0
            elif t_expr.max() > 0:
                t_expr = t_expr / t_expr.max()

            q99_act = np.quantile(t_act, 0.99)
            if q99_act > 1e-6:
                t_act = t_act / q99_act
                t_act[t_act > 1.0] = 1.0
            elif t_act.max() > 0:
                t_act = t_act / t_act.max()

            ras_vector += t_expr * t_act

        return ras_vector / len(targets)
