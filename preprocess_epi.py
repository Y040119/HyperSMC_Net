import itertools
import os
import anndata
import magic
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
import scprep
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from ot import cot_combine_sparse

DEFAULT_DB_ROOT = '/root/autodl-tmp/database'


def _resolve_db_file(is_human, filename, db_path=None):
    if db_path is not None:
        return db_path
    species_dir = 'human' if is_human else 'mouse'
    primary_path = os.path.join(DEFAULT_DB_ROOT, species_dir, filename)
    if os.path.exists(primary_path):
        return primary_path
    local_fallback = os.path.join('database', species_dir, filename)
    if os.path.exists(local_fallback):
        return local_fallback
    raise FileNotFoundError(
        f"Database file '{filename}' not found at '{primary_path}' or '{local_fallback}'. "
        f"Please pass 'db_path' explicitly."
    )


def load_lr_df(is_human=True, db_path=None):
    target_path = _resolve_db_file(is_human, 'lrpairs.tsv', db_path=db_path)
    lr_list = pd.read_csv(target_path, sep='\t')
    if 'species' in lr_list.columns:
        species = 'Human' if is_human else 'Mouse'
        lr_list = lr_list[lr_list['species'] == species]
    return lr_list


def load_pathway_df(is_human=True, db_path=None):
    target_path = _resolve_db_file(is_human, 'pathway.tsv', db_path=db_path)
    pw_list = pd.read_csv(target_path, sep='\t')
    if 'species' in pw_list.columns:
        species = 'Human' if is_human else 'Mouse'
        pw_list = pw_list[pw_list['species'] == species]
    return pw_list


def load_human_hippocampus(data_dir):
    rna_mtx_path = os.path.join(data_dir, 'Human.mtx')
    adata_rna = sc.read_mtx(rna_mtx_path).T
    genes = pd.read_csv(os.path.join(data_dir, 'genes.tsv'), header=None, sep='\t')[0].values
    barcodes_rna = pd.read_csv(os.path.join(data_dir, 'gene_barcodes.tsv'), header=None, sep='\t')[0].values
    adata_rna.var_names = genes
    adata_rna.obs_names = barcodes_rna
    adata_rna.var_names_make_unique()

    atac_mtx_path = os.path.join(data_dir, 'Human_atac.mtx')
    adata_atac = sc.read_mtx(atac_mtx_path).T
    peaks = pd.read_csv(os.path.join(data_dir, 'peaks.tsv'), header=None, sep='\t')[0].values
    barcodes_atac = pd.read_csv(os.path.join(data_dir, 'atac_barcodes.tsv'), header=None, sep='\t')[0].values
    adata_atac.var_names = peaks
    adata_atac.obs_names = barcodes_atac

    pos_path = os.path.join(data_dir, 'position.tsv')
    spatial_df = pd.read_csv(pos_path, sep=',', header=0, index_col=0)
    spatial_df.index = spatial_df.index.str.replace('"', '').str.strip()

    common_cells = np.intersect1d(adata_rna.obs_names, adata_atac.obs_names)
    common_cells = np.intersect1d(common_cells, spatial_df.index)

    adata_rna = adata_rna[common_cells, :].copy()
    adata_atac = adata_atac[common_cells, :].copy()
    spatial_df = spatial_df.loc[common_cells]

    if 'row' in spatial_df.columns and 'col' in spatial_df.columns:
        coords = spatial_df[['row', 'col']].values
    else:
        coords = spatial_df.iloc[:, 0:2].values
    adata_rna.obsm['spatial'] = coords
    adata_atac.obsm['spatial'] = coords

    return adata_rna, adata_atac


def build_gene_activity_matrix(adata_atac, gtf_path, promoter_upstream=5000, promoter_downstream=2000):
    gene_locs = {}

    with open(gtf_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if parts[2] != 'gene':
                continue

            attr_str = parts[8]
            if 'gene_name' not in attr_str:
                continue

            try:
                gene_name = attr_str.split('gene_name "')[1].split('"')[0]
            except Exception:
                continue

            chrom = parts[0]
            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]

            if strand == '+':
                p_start = start - promoter_upstream
                p_end = start + promoter_downstream
            else:
                p_start = end - promoter_downstream
                p_end = end + promoter_upstream

            if chrom not in gene_locs:
                gene_locs[chrom] = []
            gene_locs[chrom].append((p_start, p_end, gene_name))

    peak_names = adata_atac.var_names
    peak_chroms = peak_names.str.split('-').str[0]
    peak_starts = peak_names.str.split('-').str[1].astype(int)
    peak_ends = peak_names.str.split('-').str[2].astype(int)
    peak_centers = (peak_starts + peak_ends) // 2

    gene_peak_indices = {}

    for chrom, peaks_in_chrom_indices in tqdm(adata_atac.var.groupby(peak_chroms).indices.items(), desc="Processing Chromosomes"):
        if chrom not in gene_locs:
            continue
        current_centers = peak_centers[peaks_in_chrom_indices]
        current_genes = gene_locs[chrom]

        for p_start, p_end, gene in current_genes:
            match_mask = (current_centers >= p_start) & (current_centers <= p_end)
            if np.any(match_mask):
                matched_indices = peaks_in_chrom_indices[match_mask]
                if gene not in gene_peak_indices:
                    gene_peak_indices[gene] = []
                gene_peak_indices[gene].extend(matched_indices)

    unique_genes = sorted(list(gene_peak_indices.keys()))
    n_cells = adata_atac.shape[0]
    n_genes = len(unique_genes)

    activity_matrix = sparse.lil_matrix((n_cells, n_genes), dtype=np.float32)

    for i, gene in enumerate(tqdm(unique_genes, desc="Aggregating Peaks")):
        peak_idxs = gene_peak_indices[gene]
        if len(peak_idxs) > 0:
            activity_matrix[:, i] = np.array(adata_atac.X[:, peak_idxs].sum(axis=1)).flatten()

    adata_activity = anndata.AnnData(X=activity_matrix.tocsr())
    adata_activity.obs_names = adata_atac.obs_names
    adata_activity.var_names = unique_genes
    adata_activity.obsm['spatial'] = adata_atac.obsm['spatial']

    return adata_activity


def find_dual_svgs(adata_rna, adata_activity, n_top_genes=2000):
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.normalize_total(adata_activity, target_sum=1e4)
    sc.pp.log1p(adata_activity)

    try:
        sc.pp.highly_variable_genes(adata_rna, flavor='seurat_v3', n_top_genes=n_top_genes)
    except Exception:
        sc.pp.highly_variable_genes(adata_rna, n_top_genes=n_top_genes)
    rna_svgs = adata_rna.var_names[adata_rna.var['highly_variable']]

    try:
        sc.pp.highly_variable_genes(adata_activity, flavor='seurat_v3', n_top_genes=n_top_genes)
    except Exception:
        sc.pp.highly_variable_genes(adata_activity, n_top_genes=n_top_genes)
    atac_svgs = adata_activity.var_names[adata_activity.var['highly_variable']]

    adata_rna.var['is_rna_svg'] = False
    adata_rna.var.loc[rna_svgs, 'is_rna_svg'] = True
    adata_rna.var['is_atac_svg'] = False
    common_genes = np.intersect1d(adata_rna.var_names, atac_svgs)
    adata_rna.var.loc[common_genes, 'is_atac_svg'] = True
    adata_rna.var['is_core_feature'] = adata_rna.var['is_rna_svg'] | adata_rna.var['is_atac_svg']

    return adata_rna, adata_activity


def impute_MAGIC(adata):
    magic_op = magic.MAGIC(n_jobs=5, random_state=0)
    inp = adata.to_df()
    inp = scprep.normalize.library_size_normalize(inp)
    inp = scprep.transform.sqrt(inp)
    outp = magic_op.fit_transform(inp)
    adata.X = outp


def idata_construct(score, direction, pairs_meta, lr_df, lr_raw, pathway_df, adata, drop=True, normalize_total=False):
    idata = anndata.AnnData(score)
    idata.layers['direction'] = direction
    idata.obs_names = pairs_meta.index
    idata.var_names = lr_df.index
    idata.var = lr_df
    idata.uns['lr_meta'] = lr_raw
    idata.uns['pathway_meta'] = pathway_df
    idata.obs = pairs_meta
    unique_cells = np.unique(idata.obs[['A', 'B']].to_numpy().flatten())
    cell_meta = adata.obs.loc[unique_cells]
    idata.uns['cell_meta'] = cell_meta
    idata.uns['tf_count'] = adata.to_df()[np.unique(pathway_df[['src', 'dest']])].to_numpy()
    idata.uns['tf_header'] = np.unique(pathway_df[['src', 'dest']])
    sc.pp.calculate_qc_metrics(idata, inplace=True, percent_top=None)
    sc.pp.filter_genes(idata, min_cells=5)
    if drop:
        sc.pp.filter_cells(idata, min_genes=1)
    if normalize_total:
        sc.pp.normalize_total(adata)
    return idata


def subset_lr(is_human=True, db_path=None):
    lr_raw = load_lr_df(is_human=is_human, db_path=db_path).drop_duplicates(subset=['ligand', 'receptor'], keep="last")
    lr_raw['score'] = 1
    return lr_raw


def build_neighbors(row, edges):
    source_node = row.A
    target_node = row.B
    return pd.Series({
        'A': [n for n in edges.loc[source_node]['neighbor'] if n != target_node],
        'B': [n for n in edges.loc[target_node]['neighbor'] if n != source_node]
    })


def get_interface_neighbors(adata, interface_meta):
    neighborA = interface_meta.groupby('A')['B'].apply(list)
    neighborB = interface_meta.groupby('B')['A'].apply(list)
    neighbor = pd.concat([neighborA, neighborB], axis=0).reset_index(drop=False)
    neighbor.columns = ['A', 'B']
    neighbor = pd.Series(neighbor.groupby('A')['B'].sum())
    df = pd.DataFrame(index=adata.obs_names, columns=['neighbor'])
    df['neighbor'] = [[]] * len(df)
    df.loc[neighbor.index, 'neighbor'] = neighbor.to_numpy()
    return interface_meta.apply(build_neighbors, args=(df,), axis=1)


def algebraic_mean_v1(row, df, is_source, alpha=0.3):
    related_samples = row.A if is_source else row.B
    values = df.loc[related_samples]
    return values.mean() * (1 - alpha) + values.max() * alpha


def score(adata, lr_df, pairs, interface_meta):
    interface_neighbor = get_interface_neighbors(adata, interface_meta)
    exp_ref = adata.to_df()
    exp_ref = exp_ref.loc[:, ~exp_ref.columns.duplicated()]
    l = lr_df['ligand'].to_numpy().flatten()
    r = lr_df['receptor'].to_numpy().flatten()
    sub_exp = exp_ref[np.concatenate((l, r))]
    sub_exp_rev = exp_ref[np.concatenate((r, l))]
    neighbor_exp_A = interface_neighbor.apply(algebraic_mean_v1, args=(sub_exp, True), axis=1)
    neighbor_exp_B = interface_neighbor.apply(algebraic_mean_v1, args=(sub_exp_rev, False), axis=1)
    mask_A = neighbor_exp_A.isna().any(axis=1)
    mask_B = neighbor_exp_B.isna().any(axis=1)
    neighbor_exp_A[mask_A] = sub_exp.loc[interface_meta.A[mask_A]].to_numpy()
    neighbor_exp_B[mask_B] = sub_exp_rev.loc[interface_meta.B[mask_B]].to_numpy()
    neighbor_exp_A = neighbor_exp_A.to_numpy()
    neighbor_exp_B = neighbor_exp_B.to_numpy()
    sub_exp = sub_exp.to_numpy()
    sub_exp_rev = sub_exp_rev.to_numpy()
    edge_exp_both = np.multiply(sub_exp[pairs[0]] + neighbor_exp_A, sub_exp_rev[pairs[1]] + neighbor_exp_B) / 4
    res_score = np.sqrt(np.maximum(edge_exp_both[:, :int(len(l))], edge_exp_both[:, int(len(l)):]))
    direction = np.argmax((edge_exp_both[:, :int(len(l))], edge_exp_both[:, int(len(l)):]), 0)
    return res_score, direction


def score_ot(adata, lr_df, interface_meta, interface_cell_pair, weight=(0.25, 0.25, 0.25, 0.25), itermax=1000):
    data_genes = set(adata.var_names)
    ligs = list(set(lr_df.iloc[:, 0]).intersection(data_genes))
    recs = list(set(lr_df.iloc[:, 1]).intersection(data_genes))
    A = np.inf * np.ones([len(ligs), len(recs)], float)
    for i in range(len(lr_df)):
        tmp_lig = lr_df.iloc[i][0]
        tmp_rec = lr_df.iloc[i][1]
        if tmp_lig in ligs and tmp_rec in recs:
            A[ligs.index(tmp_lig), recs.index(tmp_rec)] = 1.0
    S = adata[:, ligs].to_df().to_numpy()
    D = adata[:, recs].to_df().to_numpy()

    M = np.zeros((adata.shape[0], adata.shape[0]))
    rows, cols = zip(*interface_cell_pair.T)
    M[rows, cols] = [
        np.finfo(np.float32).eps + np.linalg.norm(adata.obsm['spatial'][p[0]] - adata.obsm['spatial'][p[1]])
        for p in interface_cell_pair.T
    ]
    M += M.T
    cutoff = float(np.max(np.max(M)) * 1.1) * np.ones_like(A)
    M[M == 0] = np.max(np.max(M)) * 10

    cot_eps = 0.11
    cot_rho = 1e1

    comm_network = cot_combine_sparse(
        S, D, A, M, cutoff,
        eps_p=cot_eps, eps_mu=cot_eps, eps_nu=cot_eps, rho=cot_rho, weights=weight, nitermax=itermax
    )

    idata_score = pd.DataFrame(np.zeros((len(interface_cell_pair[0]), len(comm_network))), columns=lr_df.index)
    idata_direction = pd.DataFrame(np.ones((len(interface_cell_pair[0]), len(comm_network))), columns=lr_df.index) * -1
    for key, val in comm_network.items():
        l, r = ligs[key[0]], recs[key[1]]
        pair_name = f'{l}_{r}'
        vals = []
        dirs = []
        lsum = S[:, key[0]]
        rsum = D[:, key[1]]
        vals_coexp = []
        for pair in interface_cell_pair.T:
            coexp = np.sqrt(np.max([lsum[pair[0]] * rsum[pair[1]], lsum[pair[1]] * rsum[pair[0]]]))
            val_two = [val[pair[0], pair[1]], val[pair[1], pair[0]]]
            vals.append(np.max(val_two))
            vals_coexp.append(coexp)
            dirs.append(np.argmax(val_two))
        max_scale = np.max([np.max(vals), np.max(vals_coexp)])
        vals_coexp = (vals_coexp - np.min(vals_coexp)) / (np.max(vals_coexp) - np.min(vals_coexp))
        vals = (vals - np.min(vals)) / (np.max(vals) - np.min(vals))
        idata_score[pair_name] = np.maximum(vals_coexp, vals) * max_scale
        idata_direction[pair_name] = dirs
    return idata_score.to_numpy(), idata_direction.to_numpy(), comm_network, S, D


def score_ot_weighted(adata, lr_df, interface_meta, interface_cell_pair, weight=(0.25, 0.25, 0.25, 0.25), itermax=1000, alpha=0.01):
    ot_score, direction, comm_network, S_smth, D_smth = score_ot(adata, lr_df, interface_meta, interface_cell_pair, weight, itermax)
    coexp_score, _ = score(adata, lr_df, interface_cell_pair, interface_meta)
    return alpha * coexp_score + (1 - alpha) * ot_score, direction, comm_network, S_smth, D_smth


def find_interfaces(adata, cluster_key, lr_df, cutoff=None, is_sc=False, adata_activity=None, alpha=0.5):
    pairs = power_tri_init(adata, lr_df, adata_activity, alpha)
    pairs_meta = meta(adata, cluster_key, pairs)

    if not cutoff:
        median = np.median(pairs_meta['dist'])
        min_dist = np.min(pairs_meta['dist'])
        cutoff = np.quantile(pairs_meta['dist'], 0.75) if median == min_dist else np.quantile(pairs_meta['dist'], 0.99)

    pairs = pairs[:, pairs_meta['dist'] <= cutoff]
    pairs_meta = pairs_meta[pairs_meta['dist'] <= cutoff]

    capacity = (adata.obs['capacity'] / max(adata.obs['capacity'])).to_numpy().reshape(-1, 1)

    val, count = np.unique(pairs_meta[['A', 'B']].to_numpy().flatten(), return_counts=True)
    gmm = GaussianMixture(n_components=np.max(count), random_state=0).fit(capacity)
    gmm_labels = gmm.predict(X=capacity / max(capacity))
    gmm_labels = np.argsort(gmm.means_.flatten())[gmm_labels]
    adata.obs['n_interface'] = 0
    adata.obs.loc[val, 'n_interface'] = count
    adata.obs['ideal_n_interface'] = gmm_labels + 1
    median = np.median(pairs_meta['dist'])

    to_remove = []
    for to_prune_class in range(5):
        indices = np.argwhere(gmm_labels == to_prune_class).flatten()
        if len(indices) > 0:
            to_prune = adata.obs_names[indices]
            for i in to_prune:
                interfaces = pairs_meta[(pairs_meta['A'] == i) | (pairs_meta['B'] == i)].sort_values(by='dist')
                interfaces = interfaces[interfaces['dist'] > median]
                to_remove += list(interfaces.index[1:])

    to_remove = np.unique(to_remove)
    pairs = pairs[:, ~pairs_meta.index.isin(to_remove)]
    pairs_meta = pairs_meta.drop(to_remove)

    if not is_sc:
        self_pairs = [range(adata.shape[0]), range(adata.shape[0])]
        self_pairs_meta = meta(adata, cluster_key, self_pairs)
        pairs = np.concatenate((pairs, self_pairs), axis=1)
        pairs_meta = pd.concat([pairs_meta, self_pairs_meta], axis=0)

    return pairs, pairs_meta


def subset_adata(adata, lr_df, pathway_df, imputation, normalize_total):
    if imputation:
        impute_MAGIC(adata)

    genes = adata.var_names.tolist()
    lr_df = lr_df[lr_df['ligand'].isin(genes) & lr_df['receptor'].isin(genes)]
    lr_df.index = lr_df['ligand'] + "_" + lr_df['receptor']
    l = lr_df['ligand'].to_numpy().flatten()
    r = lr_df['receptor'].to_numpy().flatten()
    unique_lr = np.unique(np.concatenate((l, r)))
    pathway_df = subset_pathway_df(pathway_df, lr_df, adata)
    unique_tf = np.unique(pathway_df[['src', 'dest']])
    adata = adata[:, adata.var_names.isin(np.concatenate((unique_lr, unique_tf)))]
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.filter_cells(adata, min_genes=1)
    if normalize_total:
        sc.pp.normalize_total(adata)
    genes = adata.var_names.tolist()
    lr_df = lr_df[lr_df['ligand'].isin(genes) & lr_df['receptor'].isin(genes)]
    pathway_df = pathway_df[pathway_df['src'].isin(genes) & pathway_df['dest'].isin(genes)]
    return lr_df, pathway_df, adata


def power_tri_init(adata, lr_df, adata_activity=None, alpha=0.5):
    unique_lr = np.unique(lr_df[['ligand', 'receptor']].to_numpy().flatten())
    adata_exp = adata.to_df()
    adata_exp = adata_exp / adata_exp.sum(axis=0)

    if adata_activity is None:
        capacity = adata_exp[unique_lr].sum(axis=1).values
    else:
        ligands = lr_df['ligand'].unique()
        receptors = lr_df['receptor'].unique()
        valid_ligands = [g for g in ligands if g in adata_exp.columns]
        valid_receptors = [g for g in receptors if g in adata_exp.columns]

        act_df = adata_activity.to_df()
        min_val, max_val = act_df.min().min(), act_df.max().max()
        if max_val > min_val:
            act_df = (act_df - min_val) / (max_val - min_val)

        cap_l = adata_exp[valid_ligands].sum(axis=1)
        common_r = [g for g in valid_receptors if g in act_df.columns]
        other_r = [g for g in valid_receptors if g not in act_df.columns]

        r_weighted = adata_exp[common_r] * (1 + alpha * act_df[common_r])
        r_normal = adata_exp[other_r]

        cap_r = r_weighted.sum(axis=1) + r_normal.sum(axis=1)
        capacity = (cap_l + cap_r).values

    min_dist = np.min(pdist(adata.obsm['spatial'][:, ::-1]))
    adata.obs['capacity'] = capacity

    if np.max(capacity) == np.min(capacity):
        capacity[:] = 1.0
    else:
        capacity = (capacity - np.min(capacity)) / (np.max(capacity) - np.min(capacity)) + 0.01

    capacity = capacity * min_dist
    s_norm = np.sum(adata.obsm['spatial'] ** 2, axis=1) - capacity ** 2
    s_lifted = np.concatenate([adata.obsm['spatial'], s_norm[:, None]], axis=1)
    hull = ConvexHull(s_lifted)
    tri_list = tuple(
        [a, b, c] if is_ccw_triangle(adata.obsm['spatial'][a], adata.obsm['spatial'][b], adata.obsm['spatial'][c]) else [a, c, b]
        for (a, b, c), eq in zip(hull.simplices, hull.equations)
        if eq[2] <= 0
    )
    return np.unique(list(list(sorted(edge)) for tri in tri_list for edge in itertools.combinations(tri, 2)), axis=0).T


def meta(adata, cluster_key, pairs):
    pairs_meta = pd.DataFrame()
    pairs_meta['A'] = adata.obs_names[pairs[0]]
    pairs_meta['B'] = adata.obs_names[pairs[1]]
    pairs_meta[['A_row', 'A_col']] = adata.obsm['spatial'][pairs[0]]
    pairs_meta[['B_row', 'B_col']] = adata.obsm['spatial'][pairs[1]]

    if cluster_key != '' and cluster_key in adata.obs.keys():
        node_labels_text = adata.obs[cluster_key].to_numpy()
        pairs_meta['A_label'] = node_labels_text[pairs[0]].astype(str)
        pairs_meta['B_label'] = node_labels_text[pairs[1]].astype(str)
        node_labels = adata.obs[cluster_key].astype('category').cat.codes.to_numpy() + 1
        pairs_meta['A_label_int'] = node_labels[pairs[0]]
        pairs_meta['B_label_int'] = node_labels[pairs[1]]
        pairs_meta['label_1'] = pairs_meta["A_label_int"].astype(str) + pairs_meta["B_label_int"].astype(str)
        pairs_meta['label_2'] = pairs_meta["B_label_int"].astype(str) + pairs_meta["A_label_int"].astype(str)
        pairs_meta['label_int'] = pairs_meta[['label_1', 'label_2']].astype(int).max(axis=1).astype(str).astype('category')
        label_1 = pairs_meta['A_label'].astype(str) + '_' + pairs_meta['B_label'].astype(str).to_numpy()
        label_2 = pairs_meta['B_label'].astype(str) + '_' + pairs_meta['A_label'].astype(str).to_numpy()
        pick = pairs_meta[['label_1', 'label_2']].astype(int).idxmax(axis=1).to_numpy()
        text_label = [label_1[i] if x == 'label_1' else label_2[i] for i, x in enumerate(pick)]
        pairs_meta['label'] = text_label
        pairs_meta['label'] = pairs_meta['label'].astype('category')

    pairs_meta.index = pairs_meta['A'] + "_" + pairs_meta['B']

    A_pos = pairs_meta[['A_row', 'A_col']].to_numpy(dtype=float)
    B_pos = pairs_meta[['B_row', 'B_col']].to_numpy(dtype=float)
    avg_pair_pos = (A_pos + B_pos) / 2
    pairs_meta[['row', 'col']] = avg_pair_pos
    pairs_meta['dist'] = np.linalg.norm(A_pos - B_pos, axis=1)
    return pairs_meta


def subset_pathway_df(pathways, lr_df, adata):
    pathways = pathways[['src', 'dest']]
    pathways.index = pathways['src'] + '_' + pathways['dest']
    pathways = pathways[~pathways.index.isin(lr_df.index)]
    pathways = pathways[pathways['src'].isin(adata.var_names) | pathways['dest'].isin(adata.var_names)]
    return pathways.drop_duplicates()


def is_ccw_triangle(A, B, C):
    M = np.concatenate([np.stack([A, B, C]), np.ones((3, 1))], axis=1)
    return np.linalg.det(M) > 0