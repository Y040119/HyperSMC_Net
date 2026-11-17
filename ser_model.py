import pandas as pd
import numpy as np
import scanpy as sc
from scipy import sparse, stats
from tqdm import tqdm
import anndata
import warnings
from esda.moran import Moran, Moran_Local
import libpysal
from joblib import Parallel, delayed  # [新增] 引入强大的多进程并行库

class SERInferrer:
    """
    Spatial Epigenetic Regulon (SER) Inferrer.
    
    This module infers transcription factor (TF) regulons by integrating:
    1. Transcriptomic co-expression (RNA)
    2. Epigenetic accessibility (ATAC/Gene Activity)
    3. Spatial co-localization
    """
    
    def __init__(self, adata_rna, adata_activity, n_neighbors=15):
        """
        Initialize the inferrer with aligned RNA and ATAC data.
        """
        self.adata_rna = adata_rna.copy()
        self.adata_atac = adata_activity.copy()
        
        # 1. 对齐细胞 (Cells)
        common_cells = np.intersect1d(self.adata_rna.obs_names, self.adata_atac.obs_names)
        if len(common_cells) < len(self.adata_rna.obs_names):
            print(f"Aligning cells: {len(self.adata_rna.obs_names)} -> {len(common_cells)}")
            self.adata_rna = self.adata_rna[common_cells, :].copy()
            self.adata_atac = self.adata_atac[common_cells, :].copy()
        
        # 2. 预计算空间邻居 (用于平滑)
        if 'connectivities' not in self.adata_rna.obsp:
            print(f"Computing spatial neighbors (n_neighbors={n_neighbors}) for spatial constraint...")
            sc.pp.neighbors(self.adata_rna, use_rep='spatial', n_neighbors=n_neighbors)
        
        self.connectivities = self.adata_rna.obsp['connectivities']

    def load_motif_database(self, motif_file, source='custom'):
        """
        Load TF-Target prior knowledge.
        """
        print(f"Loading Motif database from {motif_file}...")
        tf_targets = {}
        try:
            if source == 'spider_pathway':
                df = pd.read_csv(motif_file, sep='\t', index_col=0)
                pairs = df[['src', 'dest']].values
            else:
                df = pd.read_csv(motif_file, sep=None, engine='python')
                pairs = df.iloc[:, 0:2].values

            # 只保留 TF 在 RNA 数据中存在的对
            valid_genes_rna = set(self.adata_rna.var_names)
            
            count = 0
            for tf, target in pairs:
                # 只要 TF 存在即可，Target 的存在性在 infer 阶段会严格检查
                if tf in valid_genes_rna:
                    if tf not in tf_targets:
                        tf_targets[tf] = set()
                    tf_targets[tf].add(target)
                    count += 1
            
            print(f"Loaded {len(tf_targets)} TFs covering {count} potential interactions.")
            return tf_targets
            
        except Exception as e:
            print(f"Error loading motif DB: {e}")
            return {}

    def _smooth_expression(self, X):
        return self.connectivities.dot(X)

    def infer(self, tf_targets_dict, 
              global_corr_th=0.3, 
              spatial_corr_th=0.2, 
              min_activity_th=0.01,
              min_targets=3,
              moran_I=0,
              n_jobs=16): # [新增参数] 默认 -1，调用服务器所有 CPU 核心
        """
        Execute the Triple-Constraint Pruning algorithm (Multiprocessing Accelerated).
        Fixed: Handles unaligned gene lists between RNA and ATAC.
        """
        print("=== 成功加载了最新修改的 Starting SER Inference (多核极速版) ===")
        print("Step 1: Pre-computing spatial smoothing...")
        known_fake_tfs = {"Gene_1140", "Gene_1141", "Gene_1142", "Gene_1143", "Gene_1144", "Gene_232"}
        # --- 探针初始化 (加在函数/模块开头) ---
        diagnostic_drop_stats = {
            "1_not_in_rna_dict": 0,
            "2_not_in_atac_dict": 0,
            "3_atac_mean_below_1e-6": 0,
            "4_correlation_too_low": 0, # 假设有共表达相关性过滤
            "5_spatial_score_too_low": 0, # 假设有空间得分过滤
            "6_other_reasons": 0
        }
        # -----------------------------------
        
        tf_drop_stats = {
            "1_未在RNA矩阵中找到": 0,
            "2_表达细胞数少于10个": 0,
            "3_未通过Morans_I空间检验": 0,
            "4_空间平滑后方差塌陷": 0,
            "5_有效的靶基因数少于门槛": 0,
            "6_TF与靶基因军团空间共振低": 0,
            "7_成功幸存进入靶基因筛选": 0
        }

        # 准备 RNA 矩阵
        if sparse.issparse(self.adata_rna.X):
            X_raw = self.adata_rna.X
        else:
            X_raw = sparse.csr_matrix(self.adata_rna.X)
            
        # [核心修复]：强制空间权重矩阵行标准化 (Row-standardization)
        w_pysal = libpysal.weights.WSP(self.connectivities).to_W()
        w_pysal.transform = 'r'  # 'r' 代表行标准化，让每一行的权重和为 1
        if 'col' in self.adata_rna.obs and 'row' in self.adata_rna.obs:
            coords = self.adata_rna.obs[['col', 'row']].values
        elif 'spatial' in self.adata_rna.obsm:
            coords = self.adata_rna.obsm['spatial']
        else:
            raise ValueError("未找到物理坐标")

        # 为莫兰指数单独构建一个与mclust一致的权重矩阵
        # 这样平滑还是用原来的n_neighbors=15，莫兰指数用k=6的二进制KNN
        w_moran = libpysal.weights.KNN.from_array(coords, k=6)
        w_moran.transform = 'r'
        
        # 使用标准化后的权重矩阵进行平滑计算，变成真正的"局部平均"
        W_sparse = w_pysal.sparse
        X_smooth = W_sparse.dot(X_raw)
        
        # 准备 ATAC 矩阵
        if sparse.issparse(self.adata_atac.X):
            X_atac = self.adata_atac.X
        else:
            X_atac = sparse.csr_matrix(self.adata_atac.X)

        # [关键修复] 分别建立 RNA 和 ATAC 的基因索引映射
        gene_list_rna = self.adata_rna.var_names.tolist()
        gene_to_idx_rna = {g: i for i, g in enumerate(gene_list_rna)}
        
        gene_list_atac = self.adata_atac.var_names.tolist()
        gene_to_idx_atac = {g: i for i, g in enumerate(gene_list_atac)}
        
        final_regulons = {}
        ras_dict = {}
        
        # =====================================================================
        # [核心重构] 定义并行计算的基础工作单元 (Worker)
        # 完好保留了您的所有注释代码与拦截逻辑
        # =====================================================================
        def _process_single_tf(tf, targets):
            local_tf_drop = {k: 0 for k in tf_drop_stats.keys()}
            local_diag_drop = {k: 0 for k in diagnostic_drop_stats.keys()}
            print_buffer = ""
            
            if tf not in gene_to_idx_rna: 
                local_tf_drop["1_未在RNA矩阵中找到"] += 1
                return tf, None, None, local_tf_drop, local_diag_drop, print_buffer
            
            tf_idx = gene_to_idx_rna[tf]
            # 强制转换为 float64，满足 Numba 的底层运算要求
            tf_expr = X_raw[:, tf_idx].toarray().flatten().astype(np.float64)
            tf_smooth = X_smooth[:, tf_idx].toarray().flatten().astype(np.float64)
            
            # 基础稀疏度检查
            if np.count_nonzero(tf_expr) < 10: 
                local_tf_drop["2_表达细胞数少于10个"] += 1
                return tf, None, None, local_tf_drop, local_diag_drop, print_buffer

            # === [全新升级：全局 + 局部的置换检验联合空间审查 (Global + Local Moran)] ===
            # 1. 全局审查 (基于经验分布的置换检验 p_sim，彻底放弃正态分布假设)
            # 使用 999 次随机打乱来构建零分布，对稀疏数据极其鲁棒
            mi_global = Moran(tf_expr, w_moran, permutations=999)
            passed_global = (mi_global.I > moran_I) and (mi_global.p_sim <= 0.05)
            
            if not passed_global:
                local_tf_drop["3_未通过Morans_I空间检验"] += 1
                return tf, None, None, local_tf_drop, local_diag_drop, print_buffer
                
#             # 2. 局部审查 (LISA 挽救机制)
#             passed_local = False
#             if not passed_global:
#                 # 只有在全局不显著时，才启动局部 Moran's I 计算，节省算力
#                 mi_local = Moran_Local(tf_expr, w_pysal, permutations=999)
                
#                 # 提取局部的 "High-High" (高表达聚集) 细胞簇，且置换检验 p 值显著
#                 # mi_local.q == 1 代表 High-High 聚集区
#                 significant_hh_cells = (mi_local.q == 1) & (mi_local.p_sim <= 0.05)
                
#                 # 挽救条件：如果有超过 15 个细胞形成了显著的高表达微环境，判定局部审查通过
#                 # 这里的 15 个细胞代表一个微小的局部组织结构，可根据数据集大小微调
#                 if np.sum(significant_hh_cells) >= 15:
#                     passed_local = True
#                     local_tf_drop["7_成功幸存进入靶基因筛选"] += 1
            
#             # 3. 综合裁定
#             if not (passed_global or passed_local):
#                 local_tf_drop["3_未通过Morans_I空间检验"] += 1
#                 return tf, None, None, local_tf_drop, local_diag_drop, print_buffer
                
#             # ====== 探头 1：空间审查监控 ======
#             if tf in known_fake_tfs:
#                 local_hh_count = np.sum(significant_hh_cells) if not passed_global else "N/A(全局已过)"
#                 print_buffer += f"\n🕵️ [假 TF 预警 - 空间关卡] {tf} 骗过了空间审查!\n"
#                 print_buffer += f"    -> Global: I值={mi_global.I:.4f}, p_sim={mi_global.p_sim:.4f} (是否通过: {passed_global})\n"
#                 if not passed_global:
#                     print_buffer += f"    -> Local: I值={mi_local.q}, p_sim={mi_local.p_sim} HH细胞数={local_hh_count} (是否通过: {passed_local})\n"
#             # =================================
            
#             # === [纯增量防线：非侵入式方差保留率 (Variance Retention) 审查] ===
#             var_raw = np.var(tf_expr)
            
#             if var_raw > 1e-8:
#                 var_smooth = np.var(tf_smooth)

#                 variance_retention = var_smooth / var_raw
                
#                 if variance_retention < 0.05:
#                     local_tf_drop["4_空间平滑后方差塌陷"] += 1
#                     return tf, None, None, local_tf_drop, local_diag_drop, print_buffer
                
#                 # ====== 探头 2：方差审查监控 ======
#                 if tf in known_fake_tfs:
#                     print_buffer += f"🕵️ [假 TF 预警 - 方差关卡] {tf} 骗过了方差审查!\n"
#                     print_buffer += f"    -> 原始方差={var_raw:.4f}, 平滑方差={var_smooth:.4f}\n"
#                     print_buffer += f"    -> 方差保留率={variance_retention:.4f} (阈值: 0.05)\n"
#                 # =================================
#             # ===============================================================
            
            # === [全新防线：基于空间 Regulon 活性的军团级审查] ===
            # 提取该 TF 所有潜在靶基因的 RNA 索引
            target_rna_indices = [gene_to_idx_rna[t] for t in targets if t in gene_to_idx_rna]
            if len(target_rna_indices) < min_targets:
                local_tf_drop["5_有效的靶基因数少于门槛"] += 1
                return tf, None, None, local_tf_drop, local_diag_drop, print_buffer
                
            # 计算“靶基因军团”的整体空间平滑表达量 (聚合大量靶基因，极大抵消了 Dropout 噪声)
            regulon_smooth = X_smooth[:, target_rna_indices].mean(axis=1).A1
            
            # 核心判断：将领 (TF) 的空间表达模式，是否与其带领的军团 (Regulon) 在空间上高度共振？
            tf_regulon_spatial_corr, _ = stats.spearmanr(tf_expr, regulon_smooth)
            
            # if np.isnan(tf_regulon_spatial_corr) or tf_regulon_spatial_corr < 0.05:
            #     local_tf_drop["6_TF与靶基因军团空间共振低"] += 1
            #     if tf in known_fake_tfs:
            #         print_buffer += f"    -> TF与靶基因军团空间Spearman相关性={tf_regulon_spatial_corr:.4f} (阈值: 0.08)\n"
            #     return tf, None, None, local_tf_drop, local_diag_drop, print_buffer
                
            # ====== 探头 3：空间共振监控 ======
            if tf in known_fake_tfs:
                print_buffer += f"🕵️ [假 TF 预警 - 共振关卡] {tf} 骗过了最终共振审查!\n"
                print_buffer += f"    -> TF与靶基因军团空间Spearman相关性={tf_regulon_spatial_corr:.4f} (阈值: 0.08)\n"
            # =================================

            # ---------------------------------------------------------
            # 插入诊断代码：排查靶基因在 RNA 和 ATAC 矩阵中的物理缺失分布
            # ---------------------------------------------------------
            missing_only_rna = 0
            missing_only_atac = 0
            missing_both = 0
            present_both = 0
            
            for t in targets:
                in_rna = t in gene_to_idx_rna
                in_atac = t in gene_to_idx_atac
                
                if in_rna and in_atac:
                    present_both += 1
                elif not in_rna and in_atac:
                    missing_only_rna += 1
                elif in_rna and not in_atac:
                    missing_only_atac += 1
                elif not in_rna and not in_atac:
                    missing_both += 1
                    
            print_buffer += f"TF [{tf}] 靶基因流失诊断:\n"
            print_buffer += f"  -> 先验总数: {len(targets)} | 双端均存在: {present_both}\n"
            print_buffer += f"  -> 仅缺失于 RNA: {missing_only_rna}\n"
            print_buffer += f"  -> 仅缺失于 ATAC: {missing_only_atac}\n"
            print_buffer += f"  -> 双端均缺失: {missing_both}\n"
            
            # 【新增：统计并打印数据交集过滤后的靶基因数量】
            # 使用列表推导式，提前计算出同时存在于 RNA 和 ATAC 矩阵中的靶基因
            present_targets = [t for t in targets if t in gene_to_idx_rna and t in gene_to_idx_atac]
            print_buffer += f"TF [{tf}]: 先验库靶基因数 = {len(targets)}, 存在于测序矩阵中的靶基因数 = {len(present_targets)}\n"
            
            # ---------------------------------------------------------
            # 修改后的靶基因筛选逻辑：保留前 90% (维持第一次修改框架)
            # ---------------------------------------------------------
            candidate_targets = {} # 第一阶段：暂存所有取得正向得分的候选靶基因
            
            for target in targets:
                # 1. 检查 Target 是否有 RNA 和 ATAC 数据
                # 拦截点 1：RNA 字典缺失
                if target not in gene_to_idx_rna:
                    local_diag_drop["1_not_in_rna_dict"] += 1 # 插入探针
                    continue
            
                # 拦截点 2：ATAC 字典缺失
                if target not in gene_to_idx_atac:
                    local_diag_drop["2_not_in_atac_dict"] += 1 # 插入探针
                    continue
                
                target_idx_rna = gene_to_idx_rna[target]
                target_idx_atac = gene_to_idx_atac[target]
                
                # --- 阶段一：过滤绝对死区 ---
                target_atac_vec = X_atac[:, target_idx_atac].toarray().flatten()
                atac_mean = target_atac_vec.mean()
                if atac_mean < 1e-6:
                    local_diag_drop["3_atac_mean_below_1e-6"] += 1 # 插入探针
                    continue

                # --- 阶段二：计算单兵相关性 ---
                target_expr = X_raw[:, target_idx_rna].toarray().flatten()
                g_corr, _ = stats.spearmanr(tf_expr, target_expr)
                if np.isnan(g_corr): g_corr = 0.0
                
                target_smooth = X_smooth[:, target_idx_rna].toarray().flatten()
                s_corr, _ = stats.spearmanr(tf_smooth, target_smooth)
                if np.isnan(s_corr): s_corr = 0.0
                
                # 空间看门人：保留底线空间共定位特征 (维持原版拦截阈值)
                if s_corr < spatial_corr_th and g_corr < global_corr_th:
                    local_diag_drop["4_correlation_too_low"] += 1
                    continue
                
                # --- 阶段三：计算表观加权的综合调控得分 ---
                # 引入伪计数 (Pseudo-count) 并开方，减缓极低活性靶基因的断崖式惩罚
                pseudo_count = 1e-6
                atac_conf = min(1.0, np.sqrt(atac_mean + pseudo_count) / np.sqrt(min_activity_th + pseudo_count))
                
                # 综合考虑空间与全局
                composite_corr = s_corr + 0.2 * max(0.0, g_corr)
                regulatory_score = atac_conf * composite_corr
                
                # 记录所有潜在的、具备正向调控得分的靶基因
                if regulatory_score > 0:
                    candidate_targets[target] = regulatory_score
            
            # --- 阶段四：保留前 90% 的动态阈值筛选 ---
            valid_targets = []
            if len(candidate_targets) > 0:
                scores = list(candidate_targets.values())
                dynamic_th = 0.02
                for tgt, score in candidate_targets.items():
                    if score >= dynamic_th:
                        valid_targets.append(tgt) 
                        
            # Finalize Regulon
            if len(valid_targets) >= min_targets:
                local_tf_drop["7_成功幸存进入靶基因筛选"] += 1
                ras_vector = self._calculate_ras(
                    valid_targets, 
                    gene_to_idx_rna, 
                    gene_to_idx_atac, 
                    X_raw, 
                    X_atac
                )
                return tf, valid_targets, ras_vector, local_tf_drop, local_diag_drop, print_buffer
            else:
                local_tf_drop["5_有效的靶基因数少于门槛"] += 1
                return tf, None, None, local_tf_drop, local_diag_drop, print_buffer

        # =====================================================================
        # 执行并行任务分发
        # =====================================================================
        print(f"🚀 启动多进程并发引擎 (调动 CPU 核心数: {'全部' if n_jobs == -1 else n_jobs})...")
        
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_process_single_tf)(tf, targets) 
            for tf, targets in tqdm(tf_targets_dict.items(), desc="Pruning Regulons (并行中)")
        )
        
        # =====================================================================
        # 收集并汇总并行计算结果
        # =====================================================================
        for tf, valid_targets, ras_vector, local_tf_drop, local_diag_drop, print_buffer in results:
            # 汇总全局探针
            for k, v in local_tf_drop.items(): tf_drop_stats[k] += v
            for k, v in local_diag_drop.items(): diagnostic_drop_stats[k] += v
            
            # 输出各核心计算时捕获的打印信息 (保证不串行乱码)
            # if print_buffer:
            #     print(print_buffer, end="")
                
            # 装载成功的调控子
            if valid_targets is not None and ras_vector is not None:
                final_regulons[tf] = valid_targets
                ras_dict[tf] = ras_vector

        # --- 探针报告打印 (统一在主线程输出) ---
        print("\n" + "="*50)
        print("🎯 [诊断探针] 靶基因拦截漏斗统计报告 🎯")
        print("="*50)
        for reason, count in diagnostic_drop_stats.items():
            if count > 0:
                print(f"  拦截原因: {reason:<25} | 剔除数量: {count}")
        print("="*50 + "\n")

        print("\n" + "="*55)
        print("🚩 [全局诊断] TF (将领) 早期拦截漏斗统计报告 🚩")
        print("="*55)
        for reason, count in tf_drop_stats.items():
            if count > 0:
                print(f"  {reason:<25} | 数量: {count}")
        print("="*55 + "\n")
        print(f"Inference Complete. Identified {len(final_regulons)} active regulons.")
        
        # Convert RAS dictionary to AnnData
        if len(ras_dict) > 0:
            ras_df = pd.DataFrame(ras_dict, index=self.adata_rna.obs_names)
            adata_ras = anndata.AnnData(X=ras_df.values, obs=self.adata_rna.obs)
            adata_ras.var_names = ras_df.columns
            adata_ras.obsm['spatial'] = self.adata_rna.obsm['spatial']
            return adata_ras, final_regulons
        else:
            warnings.warn("No valid regulons found with current thresholds.")
            return None, {}

    def _calculate_ras(self, targets, gene_to_idx_rna, gene_to_idx_atac, X_expr, X_atac):
        """
        Calculate Regulon Activity Score (RAS).
        Modified: Uses Quantile Normalization (p99) instead of MinMax.
        """
        ras_vector = np.zeros(X_expr.shape[0])
        
        for target in targets:
            # 分别获取索引
            idx_rna = gene_to_idx_rna[target]
            idx_atac = gene_to_idx_atac[target]
            
            # Get vectors (转换成稠密向量)
            t_expr = X_expr[:, idx_rna].toarray().flatten()
            t_act = X_atac[:, idx_atac].toarray().flatten()
            
            # --- 修改开始: 分位数归一化 (Quantile Normalization) ---
            
            # 1. 处理 RNA 表达 (Expression)
            # 计算 99% 分位数
            q99_expr = np.quantile(t_expr, 0.99)
            if q99_expr > 1e-6:  # 防止除以 0
                t_expr = t_expr / q99_expr
                t_expr[t_expr > 1.0] = 1.0  # 将超过 p99 的异常值截断为 1
            else:
                # 如果数据极度稀疏 (99% 都是 0)，回退到除以最大值
                if t_expr.max() > 0:
                    t_expr = t_expr / t_expr.max()

            # 2. 处理 ATAC 活性 (Activity)
            q99_act = np.quantile(t_act, 0.99)
            if q99_act > 1e-6:
                t_act = t_act / q99_act
                t_act[t_act > 1.0] = 1.0
            else:
                if t_act.max() > 0:
                    t_act = t_act / t_act.max()
            
            # --- 修改结束 ---
            
            # The core logic: A target contributes to TF activity 
            # only if it is EXPRESSED AND ACCESSIBLE.
            ras_vector += (t_expr * t_act)
            
        return ras_vector / len(targets)