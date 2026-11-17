import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import dhg
import preprocess_epi as pp_epi
import random
import os
import scipy.stats  # [新增] 用于皮尔逊相关系数计算

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子已固定为: {seed}")
    
import scipy.sparse as sp
from scipy.stats import spearmanr
from tqdm import tqdm

def filter_rtf_by_spatial_expression(adata_rna, pathway_df, min_corr=0.05, min_co_rate=0.05):
    """
    基于空间组学真实表达数据，对多跳 R-TF 通路进行共表达与空间共定位的双重严格筛选。
    
    参数:
    adata_rna: 包含全基因表达量与空间信息的 AnnData 对象 (Step 1 产物)
    pathway_df: 折叠后的多跳 R-TF 先验数据库 (包含 'src' 和 'dest' 列)
    min_corr: Spearman 相关系数的最小阈值 (默认 0.05，确保正向调控)
    min_co_rate: 空间共存率阈值 (受体表达的 Spot 中，有多少比例也表达了该 TF)
    """
    print(f"\n[HGNN Data-Driven Filter] 开始对 {len(pathway_df)} 条 R-TF 边执行空间共表达校验...")
    
    # 提取表达矩阵 (自动适配稀疏与稠密格式，提升查询效率)
    if sp.issparse(adata_rna.X):
        X_mat = adata_rna.X.tocsc()
    else:
        X_mat = np.array(adata_rna.X)
        
    gene_to_idx = {g: i for i, g in enumerate(adata_rna.var_names)}
    valid_edges = []
    
    # 记录统计信息
    stats = {'not_in_rna': 0, 'negative_corr': 0, 'low_overlap': 0, 'passed': 0}

    for _, row in tqdm(pathway_df.iterrows(), total=len(pathway_df), desc="过滤进度"):
        rec = row['src']
        tf = row['dest']
        
        # 1. 基础存在性校验
        if rec not in gene_to_idx or tf not in gene_to_idx:
            stats['not_in_rna'] += 1
            continue
            
        # 提取两个基因的表达向量
        expr_R = X_mat[:, gene_to_idx[rec]]
        expr_TF = X_mat[:, gene_to_idx[tf]]
        
        if sp.issparse(expr_R):
            expr_R = expr_R.toarray().flatten()
            expr_TF = expr_TF.toarray().flatten()
            
        # 极度稀疏防崩校验
        if np.sum(expr_R) == 0 or np.sum(expr_TF) == 0:
            stats['not_in_rna'] += 1
            continue

        # 2. 正向相关性检验 (Spearman 秩相关)
        corr, _ = spearmanr(expr_R, expr_TF)
        if np.isnan(corr) or corr < min_corr:
            stats['negative_corr'] += 1
            continue
            
        # 3. 空间共定位检验 (Co-occurrence Rate)
        # 计算在受体表达的细胞/Spot 中，有多少也表达了该转录因子
        mask_R = expr_R > 0
        mask_TF = expr_TF > 0
        intersection = np.sum(mask_R & mask_TF)
        
        # 公式: 共存率 = (R 与 TF 同时表达的 Spot 数) / (R 表达的总 Spot 数)
        co_rate = intersection / np.sum(mask_R) if np.sum(mask_R) > 0 else 0
        
        if co_rate < min_co_rate:
            stats['low_overlap'] += 1
            continue
            
        # 通过所有地狱测试，加入候选名单！
        valid_edges.append(row)
        stats['passed'] += 1

    df_filtered = pd.DataFrame(valid_edges)
    
    print("\n[HGNN 过滤总结]:")
    print(f" - 原始多跳连边: {len(pathway_df)}")
    print(f" - 因未在测序中捕获而被剔除: {stats['not_in_rna']}")
    print(f" - 因呈现负相关/无相关被剔除: {stats['negative_corr']} (抵御假阳性)")
    print(f" - 因空间极度不重合被剔除: {stats['low_overlap']} (抵御空间错位)")
    print(f" - 最终存活加入超图的高质量连边: {stats['passed']}")
    
    return df_filtered    
    

# =============================================================================
# 1. Model Architecture (Masking + Reconstruction)
# =============================================================================

class HGNNPredictor(nn.Module):
    def __init__(self, in_channels, hgnn_channels, linear_channels, use_bn=True, drop_rate=0.5):
        super().__init__()
        self.encoder_layers = nn.ModuleList()
        
        hgnn_channel_list = [in_channels] + hgnn_channels
        for i in range(1, len(hgnn_channel_list) - 1):
            self.encoder_layers.append(
                HGNNP_layer(hgnn_channel_list[i-1], hgnn_channel_list[i], use_bn=use_bn, drop_rate=drop_rate)
            )
        self.encoder_layers.append(
            HGNNP_layer(hgnn_channel_list[-2], hgnn_channel_list[-1], use_bn=use_bn, drop_rate=drop_rate, is_last=True)
        )

        final_dim = hgnn_channel_list[-1]

        # 重构解码器 (Reconstruction Decoder)
        self.recon_decoder = nn.Sequential(
            nn.Linear(final_dim, final_dim),
            nn.ELU(),
            nn.Linear(final_dim, in_channels)
        )

        self.lig_MLP = MLP_layer([final_dim] + linear_channels, use_bn=True) 
        self.rec_MLP = MLP_layer([final_dim] + linear_channels, use_bn=True)
        self.tf_MLP  = MLP_layer([final_dim] + linear_channels, use_bn=True)
        self.tg_MLP  = MLP_layer([final_dim] + linear_channels, use_bn=True)

    def encode(self, x, hg):
        for layer in self.encoder_layers:
            x = layer(x, hg)
        x = F.elu(x)
        return x

    def forward(self, x, hg, quads, edge_mask=None):
        all_embeddings = self.encode(x, hg)
        x_recon = self.recon_decoder(all_embeddings)

        lig_vec = self.lig_MLP(all_embeddings)[quads[:, 0]]
        rec_vec = self.rec_MLP(all_embeddings)[quads[:, 1]]
        tf_vec  = self.tf_MLP(all_embeddings)[quads[:, 2]]
        tg_vec  = self.tg_MLP(all_embeddings)[quads[:, 3]]

        # [修改] 使用 ReLU 激活函数替代绝对值，保证正相关生物学意义
        prob_lr = torch.cosine_similarity(lig_vec, rec_vec, dim=1)
        prob_rtf = torch.cosine_similarity(rec_vec, tf_vec, dim=1)
        prob_tftg = torch.cosine_similarity(tf_vec, tg_vec, dim=1)
        
        if edge_mask is not None:
            # 仅在推理打分阶段使用，过滤掉不合理的连接
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
            if self.bn: x = self.bn(x)
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
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                self.norms.append(nn.BatchNorm1d(layer_sizes[i+1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.use_bn: x = self.norms[i](x)
            x = F.elu(x)
        x = self.layers[-1](x)
        return x

# =============================================================================
# 2. 四元组构建 (真实数据版)
# =============================================================================

# def build_quadruplets(adata_rna, idata, regulons, pathway_df=None, min_corr=0.02, min_co_rate=0.05):
#     print("Building Full Cascade Quadruplets (L-R-TF-Target) for Real Data...")
#     gene_list = adata_rna.var_names.tolist()
#     gene_to_idx = {g: i for i, g in enumerate(gene_list)}
#     if pathway_df is None:
#         # 【清理假数据】直接调用 SPIDER 底层的真实通路库 (注意：如果跑小鼠数据请改 is_human=False)
#         # pathway_df = pp_epi.load_pathway_df(is_human=True)
#         pathway_df = pp_epi.load_pathway_df(is_human=False)
    
#     # ========================================================
#     # 【新增】：在这里插入数据驱动的严格筛选机制！
#     # min_corr=0.05 保证基础正向相关，min_co_rate=0.05 保证至少5%的空间重叠
#     # ========================================================
#     pathway_df = filter_rtf_by_spatial_expression(
#         adata_rna, 
#         pathway_df, 
#         min_corr=min_corr, 
#         min_co_rate=min_co_rate
#     )
    
#     # 如果过滤后一条路都没剩下，抛出严重警告
#     if pathway_df.empty:
#         raise ValueError("严重警告：经过表达量与空间共定位过滤后，所有 R-TF 通路均被切断。请检查数据稀疏度或调低过滤阈值！")
    
#     quads = []
#     valid_pair_set = set()

#     # 自动识别分隔符 (兼容部分数据用 ^，部分用 _ 的情况)
#     sample_name = idata.var_names[0]
#     sep = "^" if "^" in sample_name else "_"
#     print(f"Detected separator in idata: '{sep}'")

#     valid_lrs = [lr for lr in idata.var_names if sep in lr]
#     active_receptors = set([lr.split(sep)[1] for lr in valid_lrs])
#     pathway_df = pathway_df[pathway_df['src'].isin(active_receptors)]
#     print(f"Relevant biological pathway entries matching receptors: {len(pathway_df)}")

#     for lr_name in valid_lrs:
#         lig, rec = lr_name.split(sep)
#         if lig not in gene_to_idx or rec not in gene_to_idx: continue
#         l_idx = gene_to_idx[lig]
#         r_idx = gene_to_idx[rec]
        
#         candidate_tfs = pathway_df[pathway_df['src'] == rec]['dest'].unique()
#         for tf in candidate_tfs:
#             if tf in regulons and tf in gene_to_idx:
#                 tf_idx = gene_to_idx[tf]
#                 targets = regulons[tf]
#                 for target in targets:
#                     if target in gene_to_idx:
#                         t_idx = gene_to_idx[target]
#                         quads.append([l_idx, r_idx, tf_idx, t_idx])
#                         valid_pair_set.add((tf_idx, t_idx))
                        
#     quads = np.array(quads)
#     if len(quads) > 0:
#         unique_quads = np.unique(quads, axis=0)
#     else:
#         unique_quads = np.array([])
        
#     print(f"Constructed {len(unique_quads)} unique L-R-TF-Target hyperedges.")
#     return unique_quads, gene_list, valid_pair_set

def build_quadruplets(adata_rna, idata, regulons, pathway_df=None, min_corr=0.02, min_co_rate=0.05, min_rtgt_corr=0.05):
    print("Building Full Cascade Quadruplets (L-R-TF-Target) for Real Data...")
    gene_list = adata_rna.var_names.tolist()
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    
    if pathway_df is None:
        pathway_df = pp_epi.load_pathway_df(is_human=True)
    
    # 1. 独立过滤 R-TF (保持原有的空间共存与相关性过滤)
    pathway_df = filter_rtf_by_spatial_expression(
        adata_rna, 
        pathway_df, 
        min_corr=min_corr, 
        min_co_rate=min_co_rate
    )
    if pathway_df.empty:
        raise ValueError("严重警告：经过表达量与空间共定位过滤后，所有 R-TF 通路均被切断。请检查数据稀疏度或调低过滤阈值！")
    
    # 提取表达矩阵以备 R-Target 高速查询
    if sp.issparse(adata_rna.X):
        X_mat = adata_rna.X.tocsc()
    else:
        X_mat = np.array(adata_rna.X)

    quads = []
    valid_pair_set = set()

    sample_name = idata.var_names[0]
    sep = "^" if "^" in sample_name else "_"
    print(f"Detected separator in idata: '{sep}'")

    valid_lrs = [lr for lr in idata.var_names if sep in lr]
    active_receptors = set([lr.split(sep)[1] for lr in valid_lrs])
    pathway_df = pathway_df[pathway_df['src'].isin(active_receptors)]
    print(f"Relevant biological pathway entries matching receptors: {len(pathway_df)}")

    # === [新增] 用于加速计算的 R-Target 缓存字典 ===
    r_tgt_corr_cache = {}
    dropped_targets_count = 0

    for lr_name in valid_lrs:
        lig, rec = lr_name.split(sep)
        if lig not in gene_to_idx or rec not in gene_to_idx: continue
        l_idx = gene_to_idx[lig]
        r_idx = gene_to_idx[rec]
        
        # 提取受体表达向量 (同一个受体只提一次)
        expr_R = X_mat[:, r_idx]
        if sp.issparse(expr_R): expr_R = expr_R.toarray().flatten()
        
        candidate_tfs = pathway_df[pathway_df['src'] == rec]['dest'].unique()
        
        for tf in candidate_tfs:
            if tf in regulons and tf in gene_to_idx:
                tf_idx = gene_to_idx[tf]
                targets = regulons[tf]
                
                for target in targets:
                    if target in gene_to_idx:
                        t_idx = gene_to_idx[target]
                        
                        # === [新增] R-Target 跨跳共表达严格审查 ===
                        # 检查缓存，避免重复计算相同的 受体-靶基因 组合
                        if (r_idx, t_idx) not in r_tgt_corr_cache:
                            expr_TGT = X_mat[:, t_idx]
                            if sp.issparse(expr_TGT): expr_TGT = expr_TGT.toarray().flatten()
                            
                            # 计算受体与靶基因的空间相关性
                            r_tgt_corr, _ = spearmanr(expr_R, expr_TGT)
                            # 如果遇到全0向量或相关性不达标，记录为不合格 (False)
                            is_valid = not np.isnan(r_tgt_corr) and r_tgt_corr >= min_rtgt_corr
                            r_tgt_corr_cache[(r_idx, t_idx)] = is_valid
                        
                        # 裁决：如果受体激活区域，靶基因没有反应，直接切断，不加入超图
                        if not r_tgt_corr_cache[(r_idx, t_idx)]:
                            dropped_targets_count += 1
                            continue
                        # ===================================================

                        quads.append([l_idx, r_idx, tf_idx, t_idx])
                        valid_pair_set.add((tf_idx, t_idx))
                        
    quads = np.array(quads)
    if len(quads) > 0:
        unique_quads = np.unique(quads, axis=0)
    else:
        unique_quads = np.array([])
        
    print(f"--> [跨跳审查] 因受体与靶基因 (R-Target) 空间不共振，提前阻断了 {dropped_targets_count} 次无效超边生成。")
    print(f"Constructed {len(unique_quads)} unique L-R-TF-Target hyperedges.")
    
    return unique_quads, gene_list, valid_pair_set

def row_wise_shuffle(tensor):
    idx = torch.argsort(torch.rand_like(tensor), dim=1)
    shuffled_tensor = torch.gather(tensor, 1, idx)
    return shuffled_tensor

# =============================================================================
# 3. 动态演化辅助函数 (RNA共表达 + O(1)去重 + 自环剔除 + 先验限制)
# =============================================================================

def build_biological_candidate_pool(adata_rna, atac_gene_scores, quads, gene_list, prior_regulons, rna_threshold=0.1, atac_threshold=0.05, corr_threshold=0.1, pval_threshold=0.05):
    """
    真实生物学候选池挖掘：
    - 仅在先验已知、但被 ATAC Dropout 误杀的靶基因中挖掘
    - 加入了物理去重 (Hash Set) 和自环 (Self-loop) 剔除
    """
    print("\nBuilding Biological Candidate Pool (Strict Prior-Restricted)...")
    
    # O(1) 效率的快速去重查询字典
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
        
        # 筛选 RNA 表达达标，但 ATAC 信号缺失或微弱的基因
        if rna_val > rna_threshold and (np.isnan(atac_val) or atac_val <= atac_threshold):
            candidate_target_indices.append(i)
            
    if len(candidate_target_indices) == 0: return np.array([])
        
    valid_l_r_tf = np.unique(quads[:, :3], axis=0)
    candidate_pool = []
    gene_map = {i: g for i, g in enumerate(gene_list)}
    
    for lrtf in valid_l_r_tf:
        tf_idx = int(lrtf[2])
        tf_name = gene_map[tf_idx]
        tf_expr_vector = expr_mat[:, tf_idx].flatten()
        
        if np.std(tf_expr_vector) == 0: continue
        
        # 【严格限制】：获取该 TF 在真实生物学数据库中的合法下游
        allowed_targets = prior_regulons.get(tf_name, [])
            
        for target_idx in candidate_target_indices:
            target_idx = int(target_idx)
            target_name = gene_map[target_idx]
            
            # 1. 剔除不在先验知识库中的非生物学连接
            if target_name not in allowed_targets:
                continue
                
            # 2. 剔除转录因子自我调控的自环 (防止拓扑污染)
            if tf_idx == target_idx:
                continue
                
            potential_quad = (int(lrtf[0]), int(lrtf[1]), tf_idx, target_idx)
            
            # 3. 仅保留原本不在超图里的边
            if potential_quad not in existing_quads_set:
                target_expr_vector = expr_mat[:, target_idx].flatten()
                if np.std(target_expr_vector) == 0: continue
                
                corr, p_value = scipy.stats.pearsonr(tf_expr_vector, target_expr_vector)
                # 4. 共表达相关性必须达标
                if corr > corr_threshold and p_value < pval_threshold:
                    candidate_pool.append(list(potential_quad))
                
    candidate_pool = np.array(candidate_pool)
    print(f"  --> Constructed Candidate Pool with {len(candidate_pool)} TRULY NEW edges (Prior-Validated).")
    return candidate_pool    
    
# # =============================================================================
# # 2. Data Preparation & Utility Functions
# # =============================================================================

# def build_quadruplets(adata_rna, idata, regulons):
#     print("Building Full Cascade Quadruplets (L-R-TF-Target)...")
#     gene_list = adata_rna.var_names.tolist()
#     gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    
#     pathway_path = '/root/autodl-tmp/SPIDER/EpiSPIDER_Data/fake_pathways.tsv'
#     if os.path.exists(pathway_path):
#         print(f"Loading simulated pathway database from: {pathway_path}")
#         pathway_df = pd.read_csv(pathway_path, sep='\t')
#     else:
#         print("Warning: Simulated pathway file not found, falling back to default.")
#         pathway_df = pp_epi.load_pathway_df(is_human=True)
#         # pathway_df = pp_epi.load_pathway_df(is_human=False)
    
#     quads = []
#     valid_pair_set = set()

#     sample_name = idata.var_names[0]
#     sep = "^" if "^" in sample_name else "_"
    
#     valid_lrs = [lr for lr in idata.var_names if sep in lr]
#     active_receptors = set([lr.split(sep)[1] for lr in valid_lrs])
#     pathway_df = pathway_df[pathway_df['src'].isin(active_receptors)]

#     for lr_name in valid_lrs:
#         lig, rec = lr_name.split(sep)
#         if lig not in gene_to_idx or rec not in gene_to_idx: continue
#         l_idx = gene_to_idx[lig]
#         r_idx = gene_to_idx[rec]
        
#         candidate_tfs = pathway_df[pathway_df['src'] == rec]['dest'].unique()
        
#         for tf in candidate_tfs:
#             if tf in regulons and tf in gene_to_idx:
#                 tf_idx = gene_to_idx[tf]
#                 targets = regulons[tf]
#                 for target in targets:
#                     if target in gene_to_idx:
#                         t_idx = gene_to_idx[target]
#                         quads.append([l_idx, r_idx, tf_idx, t_idx])
#                         valid_pair_set.add((tf_idx, t_idx))
                        
#     quads = np.array(quads)
#     if len(quads) > 0:
#         unique_quads = np.unique(quads, axis=0)
#     else:
#         unique_quads = np.array([])
        
#     print(f"Constructed {len(unique_quads)} unique L-R-TF-Target hyperedges.")
#     return unique_quads, gene_list, valid_pair_set

# def get_batch_edge_mask(quads_batch, valid_pair_set, device):
#     batch_np = quads_batch.detach().cpu().numpy()
#     mask_list = []
#     for i in range(len(batch_np)):
#         tf_idx = batch_np[i, 2]
#         tg_idx = batch_np[i, 3]
#         if (tf_idx, tg_idx) in valid_pair_set:
#             mask_list.append(1.0)
#         else:
#             mask_list.append(0.0)
#     return torch.tensor(mask_list, dtype=torch.float32).to(device)

# def row_wise_shuffle(tensor):
#     idx = torch.argsort(torch.rand_like(tensor), dim=1)
#     shuffled_tensor = torch.gather(tensor, 1, idx)
#     return shuffled_tensor


# =============================================================================
# 4. Training Loop (Dynamic Graph Evolution + MAE)
# =============================================================================

# 【修改注】: 在参数列表中增加了 atac_scores_dict 参数
def run_sigxtalk_scoring(adata_rna, idata, ras_df, regulons, device='cpu', epochs=50, atac_scores_dict=None, pathway_df=None, min_corr=0.02, min_co_rate=0.05, min_rtgt_corr=0.05):
    setup_seed(42)
    quads, gene_list, valid_pair_set = build_quadruplets(adata_rna, idata, regulons, pathway_df=pathway_df, min_corr=min_corr, min_co_rate=min_co_rate, min_rtgt_corr=min_rtgt_corr)
    
    if len(quads) == 0: 
        return pd.DataFrame(columns=['Ligand', 'Receptor', 'TF', 'Target', 'SigXTalk_Score'])

    # 导出未训练的超图结构
    gene_map = {i: g for i, g in enumerate(gene_list)}
    hg_df = pd.DataFrame([{'Ligand': gene_map[q[0]], 'Receptor': gene_map[q[1]], 'TF': gene_map[q[2]], 'Target': gene_map[q[3]]} for q in quads])
    export_path = '/root/autodl-tmp/SPIDER/output_moni/untrained_hypergraph.csv'
    # 确保目录存在
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    hg_df.to_csv(export_path, index=False)
    
    # 构建生物学候选超边池
    candidate_pool = build_biological_candidate_pool(
        adata_rna=adata_rna, 
        atac_gene_scores=atac_scores_dict,  # 传入外部准备好的 ATAC 分数字典
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

    # 表达矩阵预处理
    if hasattr(adata_rna.X, 'toarray'):
        exp_matrix = adata_rna.X.toarray().T
    else:
        exp_matrix = adata_rna.X.T
    mean = exp_matrix.mean(axis=1, keepdims=True)
    std = exp_matrix.std(axis=1, keepdims=True) + 1e-6
    exp_matrix = (exp_matrix - mean) / std
    exp_tensor = torch.tensor(exp_matrix, dtype=torch.float32).to(device)
    
    # 初始化超图对象
    edge_list = [tuple(q) for q in active_quads_list]
    hg = dhg.Hypergraph(len(gene_list), edge_list).to(device)

    model = HGNNPredictor(
        in_channels=exp_tensor.shape[1], 
        hgnn_channels=[256, 128],
        linear_channels=[64, 32]
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # ... [前面保留你的代码：模型初始化，生成 candidate_pool 等] ...

    print(f"Starting Training with Skeptical Graph Evolution for {epochs} epochs...")
    quad_tensor = torch.tensor(active_quads_list, dtype=torch.long).to(device)
    num_samples = quad_tensor.size(0)
    
    # ==========================================
    # 【新增 1】：初始化置信度列表
    # 先验库里的边，我们给 100% (1.0) 的信任
    # ==========================================
    prior_edges_count = len(quads)
    edge_confidences = [1.0] * prior_edges_count
    
    update_freq = 10      
    add_threshold = 0.85  
    
    # 置信度演化参数
    init_confidence = 0.2  # 新边入局时的初始“低置信度”
    reward_step = 0.15     # 每次拿到高分，信任度增加量
    punish_step = 0.20     # 每次表现拉胯，信任度扣除量 (惩罚可以比奖励稍微严厉一点)

    model.train()
    for epoch in range(epochs):
        
        # ==========================================
        # [阶段 A] 动态图重构 (周期性新增超边 & 信任度奖惩)
        # ==========================================
        if epoch > 0 and epoch % update_freq == 0:
            model.eval() 
            with torch.no_grad():
                
                # ------------------------------------------
                # 第一步：先考核图内已有的边 (执行信任度奖惩与剔除)
                # ------------------------------------------
                current_scores, _ = model(exp_tensor, hg, quad_tensor, edge_mask=None)
                current_scores = current_scores.cpu().numpy().flatten()
                
                keep_indices = []
                for idx in range(len(active_quads_list)):
                    # 先验边拥有免死金牌，信任度永远是 1.0
                    if idx < prior_edges_count:
                        keep_indices.append(idx)
                        continue
                        
                    # 对新边执行奖惩
                    score = current_scores[idx]
                    if score > 0.80:
                        edge_confidences[idx] = min(1.0, edge_confidences[idx] + reward_step)
                    elif score < 0.50:
                        edge_confidences[idx] = max(0.0, edge_confidences[idx] - punish_step)
                        
                    # 如果信任度还没有彻底归零，准许它继续留在图中
                    if edge_confidences[idx] > 0.0:
                        keep_indices.append(idx)
                
                # 执行剔除 (Pruning)
                if len(keep_indices) < len(active_quads_list):
                    removed_count = len(active_quads_list) - len(keep_indices)
                    print(f"  --> [Pruning] {removed_count} edges lost all trust and were KICKED OUT of the graph!")
                    active_quads_list = [active_quads_list[i] for i in keep_indices]
                    edge_confidences = [edge_confidences[i] for i in keep_indices]

                # ------------------------------------------
                # 第二步：考核候选池中的未入局边 (招募新边)
                # ------------------------------------------
                if candidate_tensor is not None and len(candidate_pool) > 0:
                    cand_scores, _ = model(exp_tensor, hg, candidate_tensor, edge_mask=None)
                    cand_scores = cand_scores.cpu().numpy().flatten()
                    
                    new_idx = np.where(cand_scores > add_threshold)[0]
                    if len(new_idx) > 0:
                        print(f"--- Epoch {epoch}: Suspiciously adding {len(new_idx)} new Target genes... ---")
                        # 招募新边进入活跃列表
                        active_quads_list.extend(candidate_pool[new_idx].tolist())
                        # 给这些新加入的边赋予极低的初始置信度
                        edge_confidences.extend([init_confidence] * len(new_idx))
                        
                        # 从候选池中删去已加入的边
                        candidate_pool = np.delete(candidate_pool, new_idx, axis=0)
                        candidate_tensor = torch.tensor(candidate_pool, dtype=torch.long).to(device) if len(candidate_pool) > 0 else None
                
                # ------------------------------------------
                # 第三步：重新实例化底层超图结构
                # ------------------------------------------
                new_edge_list = [tuple(q) for q in active_quads_list]
                hg = dhg.Hypergraph(len(gene_list), new_edge_list).to(device)
                
                # 重新生成训练 Tensor，准备进入下一轮
                quad_tensor = torch.tensor(active_quads_list, dtype=torch.long).to(device)
                num_samples = quad_tensor.size(0)
            
            model.train()

        # ==========================================
        # [阶段 B] 常规训练 (将置信度融入 Loss)
        # ==========================================
        optimizer.zero_grad()
        
        mask_idx = torch.randperm(exp_tensor.size(0))[:int(exp_tensor.size(0) * 0.2)].to(device)
        masked_exp = exp_tensor.clone()
        masked_exp[mask_idx] = 0.0 
        
        # 1. Positive Samples 
        pred_pos, x_recon = model(masked_exp, hg, quad_tensor, edge_mask=None)
        
        # 【新增 3】：带有“怀疑机制”的加权 Loss
        # 将我们维护的 list 转为 tensor，准备去给 Loss 当权重
        conf_tensor = torch.tensor(edge_confidences, dtype=torch.float32).to(device).view(-1, 1)
        
        # 注意：此处必须用 reduction='none'，才能拿到每条边独立的 Loss，再去乘以权重
        loss_pos_unweighted = F.binary_cross_entropy(pred_pos, torch.ones_like(pred_pos), reduction='none')
        loss_pos = (loss_pos_unweighted * conf_tensor).mean() # 加权后求均值

        # 2. Easy Negatives (保持原样)
        pred_neg_easy, _ = model(row_wise_shuffle(exp_tensor), hg, quad_tensor, edge_mask=None)
        loss_neg_easy = F.binary_cross_entropy(pred_neg_easy, torch.zeros_like(pred_neg_easy))
        
        # 3. Hard Negatives (保持原样)
        quad_neg_hard = quad_tensor.clone()
        quad_neg_hard[:, 3] = quad_neg_hard[torch.randperm(quad_tensor.size(0)), 3] 
        pred_neg_hard, _ = model(masked_exp, hg, quad_neg_hard, edge_mask=None)
        loss_neg_hard = F.binary_cross_entropy(pred_neg_hard, torch.zeros_like(pred_neg_hard))
        
        # 新增的重构 Loss
        loss_recon = F.mse_loss(x_recon[mask_idx], exp_tensor[mask_idx])
        
        loss = loss_pos + 0.5 * loss_neg_easy + 0.5 * loss_neg_hard + 0.1 * loss_recon
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            # 统计当前有多少“完全信任”的新边
            trusted_new = sum(1 for c in edge_confidences[prior_edges_count:] if c >= 0.95)
            total_new = len(edge_confidences) - prior_edges_count
            print(f"Epoch {epoch+1} | Total Loss: {loss.item():.4f} | New Edges: {total_new} ({trusted_new} fully trusted)")

    # ... [后面的 Scoring 和 Output 代码保持你现有的版本不变] ...
    # ==========================================
    # 5. 推理阶段 (应用验证库的 Mask)
    # ==========================================
    print("\nScoring Final Output...")
    model.eval()
    with torch.no_grad():
        # 【关键保护机制】：在这里，我们需要对 active_quads_list 进行判断
        # 如果是新加的进化边，我们允许它通过；如果是原本就在 quads 里但不合法的边，仍然 Mask 掉。
        # 为了兼容，我们将所有在 active_quads_list 里的视为合法
        # 因为我们的 candidate_pool 是我们精挑细选出来的。
        mask_final = torch.ones(num_samples, dtype=torch.float32).to(device)
        
        final_scores, _ = model(exp_tensor, hg, quad_tensor, edge_mask=mask_final)
        final_scores = final_scores.cpu().numpy().flatten()
    
    # --- Output ---
    results = []
    for idx, (l_i, r_i, tf_i, tg_i) in enumerate(active_quads_list):
        results.append({
            'Ligand': gene_map[l_i], 'Receptor': gene_map[r_i],
            'TF': gene_map[tf_i], 'Target': gene_map[tg_i],
            'SigXTalk_Score': final_scores[idx],
            # 打上标记，区分哪些是先验已知，哪些是网络自主发现的
            'Is_Discovered': 'Yes' if idx >= len(quads) else 'No' 
        })
        
    return pd.DataFrame(results).sort_values(by='SigXTalk_Score', ascending=False)