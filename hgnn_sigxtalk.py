import random
import dhg
import numpy as np
import pandas as pd
import preprocess_epi as pp_epi
import scipy.sparse as sp
import scipy.stats
from scipy.stats import spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def filter_rtf_by_spatial_expression(adata_rna, pathway_df, min_corr=0.05, min_co_rate=0.05):
    if sp.issparse(adata_rna.X):
        X_mat = adata_rna.X.tocsc()
    else:
        X_mat = np.array(adata_rna.X)

    gene_to_idx = {g: i for i, g in enumerate(adata_rna.var_names)}
    valid_edges = []

    for _, row in tqdm(pathway_df.iterrows(), total=len(pathway_df), desc="Filtering R-TF pairs"):
        rec = row['src']
        tf = row['dest']

        if rec not in gene_to_idx or tf not in gene_to_idx:
            continue

        expr_R = X_mat[:, gene_to_idx[rec]]
        expr_TF = X_mat[:, gene_to_idx[tf]]

        if sp.issparse(expr_R):
            expr_R = expr_R.toarray().flatten()
            expr_TF = expr_TF.toarray().flatten()

        if np.sum(expr_R) == 0 or np.sum(expr_TF) == 0:
            continue

        corr, _ = spearmanr(expr_R, expr_TF)
        if np.isnan(corr) or corr < min_corr:
            continue

        mask_R = expr_R > 0
        mask_TF = expr_TF > 0
        intersection = np.sum(mask_R & mask_TF)
        co_rate = intersection / np.sum(mask_R) if np.sum(mask_R) > 0 else 0

        if co_rate < min_co_rate:
            continue

        valid_edges.append(row)

    return pd.DataFrame(valid_edges)


class HGNNPredictor(nn.Module):
    def __init__(self, in_channels, hgnn_channels, linear_channels, use_bn=True, drop_rate=0.5):
        super().__init__()
        self.encoder_layers = nn.ModuleList()

        hgnn_channel_list = [in_channels] + hgnn_channels
        for i in range(1, len(hgnn_channel_list) - 1):
            self.encoder_layers.append(
                HGNNP_layer(hgnn_channel_list[i - 1], hgnn_channel_list[i], use_bn=use_bn, drop_rate=drop_rate)
            )
        self.encoder_layers.append(
            HGNNP_layer(hgnn_channel_list[-2], hgnn_channel_list[-1], use_bn=use_bn, drop_rate=drop_rate, is_last=True)
        )

        final_dim = hgnn_channel_list[-1]

        self.recon_decoder = nn.Sequential(
            nn.Linear(final_dim, final_dim),
            nn.ELU(),
            nn.Linear(final_dim, in_channels)
        )

        self.lig_MLP = MLP_layer([final_dim] + linear_channels, use_bn=True)
        self.rec_MLP = MLP_layer([final_dim] + linear_channels, use_bn=True)
        self.tf_MLP = MLP_layer([final_dim] + linear_channels, use_bn=True)
        self.tg_MLP = MLP_layer([final_dim] + linear_channels, use_bn=True)

    def encode(self, x, hg):
        for layer in self.encoder_layers:
            x = layer(x, hg)
        return F.elu(x)

    def forward(self, x, hg, quads, edge_mask=None):
        all_embeddings = self.encode(x, hg)
        x_recon = self.recon_decoder(all_embeddings)

        lig_vec = self.lig_MLP(all_embeddings)[quads[:, 0]]
        rec_vec = self.rec_MLP(all_embeddings)[quads[:, 1]]
        tf_vec = self.tf_MLP(all_embeddings)[quads[:, 2]]
        tg_vec = self.tg_MLP(all_embeddings)[quads[:, 3]]

        prob_lr = torch.cosine_similarity(lig_vec, rec_vec, dim=1)
        prob_rtf = torch.cosine_similarity(rec_vec, tf_vec, dim=1)
        prob_tftg = torch.cosine_similarity(tf_vec, tg_vec, dim=1)

        if edge_mask is not None:
            prob_tftg = prob_tftg * edge_mask

        prob = F.relu(prob_lr * prob_rtf * prob_tftg)

        return prob.view(-1, 1), x_recon


class HGNNP_layer(nn.Module):
    def __init__(self, in_channels, out_channels, leaky_alpha=0.1, use_bn=True, drop_rate=0.5, is_last=False):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.LeakyReLU(negative_slope=leaky_alpha)
        self.theta = nn.Linear(in_channels, out_channels)
        self.drop_rate = drop_rate

    def forward(self, x, hg):
        x = self.theta(x)
        x = hg.v2v(x, aggr="mean")
        if not self.is_last:
            x = self.act(x)
            if self.bn:
                x = self.bn(x)
            x = F.dropout(x, p=self.drop_rate, training=self.training)
            x = F.normalize(x, p=2, dim=1)
        return x


class MLP_layer(nn.Module):
    def __init__(self, layer_sizes, use_bn=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_bn = use_bn
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                self.norms.append(nn.BatchNorm1d(layer_sizes[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.use_bn:
                x = self.norms[i](x)
            x = F.elu(x)
        return self.layers[-1](x)


def build_quadruplets(adata_rna, idata, regulons, pathway_df=None, min_corr=0.02, min_co_rate=0.05, min_rtgt_corr=0.05):
    gene_list = adata_rna.var_names.tolist()
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    if pathway_df is None:
        pathway_df = pp_epi.load_pathway_df(is_human=True)

    pathway_df = filter_rtf_by_spatial_expression(
        adata_rna,
        pathway_df,
        min_corr=min_corr,
        min_co_rate=min_co_rate
    )
    if pathway_df.empty:
        raise ValueError("No R-TF pathways remained after spatial expression and co-localization filtering.")

    if sp.issparse(adata_rna.X):
        X_mat = adata_rna.X.tocsc()
    else:
        X_mat = np.array(adata_rna.X)

    quads = []
    valid_pair_set = set()

    sample_name = idata.var_names[0]
    sep = "^" if "^" in sample_name else "_"

    valid_lrs = [lr for lr in idata.var_names if sep in lr]
    active_receptors = set([lr.split(sep)[1] for lr in valid_lrs])
    pathway_df = pathway_df[pathway_df['src'].isin(active_receptors)]

    r_tgt_corr_cache = {}

    for lr_name in valid_lrs:
        lig, rec = lr_name.split(sep)
        if lig not in gene_to_idx or rec not in gene_to_idx:
            continue
        l_idx = gene_to_idx[lig]
        r_idx = gene_to_idx[rec]

        expr_R = X_mat[:, r_idx]
        if sp.issparse(expr_R):
            expr_R = expr_R.toarray().flatten()

        candidate_tfs = pathway_df[pathway_df['src'] == rec]['dest'].unique()

        for tf in candidate_tfs:
            if tf in regulons and tf in gene_to_idx:
                tf_idx = gene_to_idx[tf]
                targets = regulons[tf]

                for target in targets:
                    if target in gene_to_idx:
                        t_idx = gene_to_idx[target]

                        if (r_idx, t_idx) not in r_tgt_corr_cache:
                            expr_TGT = X_mat[:, t_idx]
                            if sp.issparse(expr_TGT):
                                expr_TGT = expr_TGT.toarray().flatten()

                            r_tgt_corr, _ = spearmanr(expr_R, expr_TGT)
                            is_valid = not np.isnan(r_tgt_corr) and r_tgt_corr >= min_rtgt_corr
                            r_tgt_corr_cache[(r_idx, t_idx)] = is_valid

                        if not r_tgt_corr_cache[(r_idx, t_idx)]:
                            continue

                        quads.append([l_idx, r_idx, tf_idx, t_idx])
                        valid_pair_set.add((tf_idx, t_idx))

    quads = np.array(quads)
    unique_quads = np.unique(quads, axis=0) if len(quads) > 0 else np.array([])

    return unique_quads, gene_list, valid_pair_set


def row_wise_shuffle(tensor):
    idx = torch.argsort(torch.rand_like(tensor), dim=1)
    return torch.gather(tensor, 1, idx)


def build_biological_candidate_pool(adata_rna, atac_gene_scores, quads, gene_list, prior_regulons, rna_threshold=0.1, atac_threshold=0.05, corr_threshold=0.1, pval_threshold=0.05):
    existing_quads_set = set(map(tuple, quads))

    if hasattr(adata_rna.X, 'toarray'):
        expr_mat = adata_rna.X.toarray()
    else:
        expr_mat = adata_rna.X
    rna_expr_mean = np.array(expr_mat.mean(axis=0)).flatten()

    candidate_target_indices = []
    is_dict = isinstance(atac_gene_scores, dict)

    for i, gene in enumerate(gene_list):
        rna_val = rna_expr_mean[i]
        if atac_gene_scores is not None:
            atac_val = atac_gene_scores.get(gene, 0.0) if is_dict else atac_gene_scores[i]
        else:
            atac_val = 0.0

        if rna_val > rna_threshold and (np.isnan(atac_val) or atac_val <= atac_threshold):
            candidate_target_indices.append(i)

    if len(candidate_target_indices) == 0:
        return np.array([])

    valid_l_r_tf = np.unique(quads[:, :3], axis=0)
    candidate_pool = []
    gene_map = {i: g for i, g in enumerate(gene_list)}

    for lrtf in valid_l_r_tf:
        tf_idx = int(lrtf[2])
        tf_name = gene_map[tf_idx]
        tf_expr_vector = expr_mat[:, tf_idx].flatten()

        if np.std(tf_expr_vector) == 0:
            continue

        allowed_targets = prior_regulons.get(tf_name, [])

        for target_idx in candidate_target_indices:
            target_idx = int(target_idx)
            target_name = gene_map[target_idx]

            if target_name not in allowed_targets or tf_idx == target_idx:
                continue

            potential_quad = (int(lrtf[0]), int(lrtf[1]), tf_idx, target_idx)

            if potential_quad not in existing_quads_set:
                target_expr_vector = expr_mat[:, target_idx].flatten()
                if np.std(target_expr_vector) == 0:
                    continue

                corr, p_value = scipy.stats.pearsonr(tf_expr_vector, target_expr_vector)
                if corr > corr_threshold and p_value < pval_threshold:
                    candidate_pool.append(list(potential_quad))

    return np.array(candidate_pool)


def run_sigxtalk_scoring(adata_rna, idata, ras_df, regulons, device='cpu', epochs=50, atac_scores_dict=None, pathway_df=None, min_corr=0.02, min_co_rate=0.05, min_rtgt_corr=0.05):
    setup_seed(42)
    quads, gene_list, _ = build_quadruplets(
        adata_rna,
        idata,
        regulons,
        pathway_df=pathway_df,
        min_corr=min_corr,
        min_co_rate=min_co_rate,
        min_rtgt_corr=min_rtgt_corr
    )

    if len(quads) == 0:
        return pd.DataFrame(columns=['Ligand', 'Receptor', 'TF', 'Target', 'SigXTalk_Score', 'Is_Discovered'])

    gene_map = {i: g for i, g in enumerate(gene_list)}

    candidate_pool = build_biological_candidate_pool(
        adata_rna=adata_rna,
        atac_gene_scores=atac_scores_dict,
        quads=quads,
        gene_list=gene_list,
        prior_regulons=regulons,
        rna_threshold=0.1,
        atac_threshold=0.05,
        corr_threshold=0.1,
        pval_threshold=0.05
    )
    candidate_tensor = torch.tensor(candidate_pool, dtype=torch.long).to(device) if len(candidate_pool) > 0 else None
    active_quads_list = quads.tolist()

    if hasattr(adata_rna.X, 'toarray'):
        exp_matrix = adata_rna.X.toarray().T
    else:
        exp_matrix = adata_rna.X.T
    mean = exp_matrix.mean(axis=1, keepdims=True)
    std = exp_matrix.std(axis=1, keepdims=True) + 1e-6
    exp_matrix = (exp_matrix - mean) / std
    exp_tensor = torch.tensor(exp_matrix, dtype=torch.float32).to(device)

    edge_list = [tuple(q) for q in active_quads_list]
    hg = dhg.Hypergraph(len(gene_list), edge_list).to(device)

    model = HGNNPredictor(
        in_channels=exp_tensor.shape[1],
        hgnn_channels=[256, 128],
        linear_channels=[64, 32]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    quad_tensor = torch.tensor(active_quads_list, dtype=torch.long).to(device)
    num_samples = quad_tensor.size(0)

    prior_edges_count = len(quads)
    edge_confidences = [1.0] * prior_edges_count

    update_freq = 10
    add_threshold = 0.85
    init_confidence = 0.2
    reward_step = 0.15
    punish_step = 0.20

    model.train()
    for epoch in range(epochs):
        if epoch > 0 and epoch % update_freq == 0:
            model.eval()
            with torch.no_grad():
                current_scores, _ = model(exp_tensor, hg, quad_tensor, edge_mask=None)
                current_scores = current_scores.cpu().numpy().flatten()

                keep_indices = []
                for idx in range(len(active_quads_list)):
                    if idx < prior_edges_count:
                        keep_indices.append(idx)
                        continue

                    score = current_scores[idx]
                    if score > 0.80:
                        edge_confidences[idx] = min(1.0, edge_confidences[idx] + reward_step)
                    elif score < 0.50:
                        edge_confidences[idx] = max(0.0, edge_confidences[idx] - punish_step)

                    if edge_confidences[idx] > 0.0:
                        keep_indices.append(idx)

                if len(keep_indices) < len(active_quads_list):
                    active_quads_list = [active_quads_list[i] for i in keep_indices]
                    edge_confidences = [edge_confidences[i] for i in keep_indices]

                if candidate_tensor is not None and len(candidate_pool) > 0:
                    cand_scores, _ = model(exp_tensor, hg, candidate_tensor, edge_mask=None)
                    cand_scores = cand_scores.cpu().numpy().flatten()

                    new_idx = np.where(cand_scores > add_threshold)[0]
                    if len(new_idx) > 0:
                        active_quads_list.extend(candidate_pool[new_idx].tolist())
                        edge_confidences.extend([init_confidence] * len(new_idx))

                        candidate_pool = np.delete(candidate_pool, new_idx, axis=0)
                        candidate_tensor = torch.tensor(candidate_pool, dtype=torch.long).to(device) if len(candidate_pool) > 0 else None

                new_edge_list = [tuple(q) for q in active_quads_list]
                hg = dhg.Hypergraph(len(gene_list), new_edge_list).to(device)
                quad_tensor = torch.tensor(active_quads_list, dtype=torch.long).to(device)
                num_samples = quad_tensor.size(0)

            model.train()

        optimizer.zero_grad()

        mask_idx = torch.randperm(exp_tensor.size(0))[:int(exp_tensor.size(0) * 0.2)].to(device)
        masked_exp = exp_tensor.clone()
        masked_exp[mask_idx] = 0.0

        pred_pos, x_recon = model(masked_exp, hg, quad_tensor, edge_mask=None)

        conf_tensor = torch.tensor(edge_confidences, dtype=torch.float32).to(device).view(-1, 1)
        loss_pos_unweighted = F.binary_cross_entropy(pred_pos, torch.ones_like(pred_pos), reduction='none')
        loss_pos = (loss_pos_unweighted * conf_tensor).mean()

        pred_neg_easy, _ = model(row_wise_shuffle(exp_tensor), hg, quad_tensor, edge_mask=None)
        loss_neg_easy = F.binary_cross_entropy(pred_neg_easy, torch.zeros_like(pred_neg_easy))

        quad_neg_hard = quad_tensor.clone()
        quad_neg_hard[:, 3] = quad_neg_hard[torch.randperm(quad_tensor.size(0)), 3]
        pred_neg_hard, _ = model(masked_exp, hg, quad_neg_hard, edge_mask=None)
        loss_neg_hard = F.binary_cross_entropy(pred_neg_hard, torch.zeros_like(pred_neg_hard))

        loss_recon = F.mse_loss(x_recon[mask_idx], exp_tensor[mask_idx])

        loss = loss_pos + 0.5 * loss_neg_easy + 0.5 * loss_neg_hard + 0.1 * loss_recon
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        mask_final = torch.ones(num_samples, dtype=torch.float32).to(device)
        final_scores, _ = model(exp_tensor, hg, quad_tensor, edge_mask=mask_final)
        final_scores = final_scores.cpu().numpy().flatten()

    results = []
    for idx, (l_i, r_i, tf_i, tg_i) in enumerate(active_quads_list):
        results.append({
            'Ligand': gene_map[l_i],
            'Receptor': gene_map[r_i],
            'TF': gene_map[tf_i],
            'Target': gene_map[tg_i],
            'SigXTalk_Score': final_scores[idx],
            'Is_Discovered': 'Yes' if idx >= len(quads) else 'No'
        })

    return pd.DataFrame(results).sort_values(by='SigXTalk_Score', ascending=False)
