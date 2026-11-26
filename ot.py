import numpy as np
from scipy import sparse


def cot_combine_sparse(S, D, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, weights=(0.1, 0.1, 0.1, 0.7), nitermax=1e4, stopthr=1e-8):
    if isinstance(eps_p, tuple):
        eps_p_cot, eps_p_row, eps_p_col, eps_p_blk = eps_p
    else:
        eps_p_cot = eps_p_row = eps_p_col = eps_p_blk = eps_p
    if isinstance(rho, tuple):
        rho_cot, rho_row, rho_col, rho_blk = rho
    else:
        rho_cot = rho_row = rho_col = rho_blk = rho
    if eps_mu is None:
        eps_mu_cot = eps_p_cot
        eps_mu_row = eps_p_row
        eps_mu_col = eps_p_col
        eps_mu_blk = eps_p_blk
    elif isinstance(eps_mu, tuple):
        eps_mu_cot, eps_mu_row, eps_mu_col, eps_mu_blk = eps_mu
    else:
        eps_mu_cot = eps_mu_row = eps_mu_col = eps_mu_blk = eps_mu
    if eps_nu is None:
        eps_nu_cot = eps_p_cot
        eps_nu_row = eps_p_row
        eps_nu_col = eps_p_col
        eps_nu_blk = eps_p_blk
    elif isinstance(eps_nu, tuple):
        eps_nu_cot, eps_nu_row, eps_nu_col, eps_nu_blk = eps_nu
    else:
        eps_nu_cot = eps_nu_row = eps_nu_col = eps_nu_blk = eps_nu

    P_cot = cot_sparse(
        S, D, A, M, cutoff,
        eps_p=eps_p_cot, eps_mu=eps_mu_cot, eps_nu=eps_nu_cot, rho=rho_cot,
        nitermax=nitermax, stopthr=stopthr
    )
    P_row = cot_row_sparse(
        S, D, A, M, cutoff,
        eps_p=eps_p_row, eps_mu=eps_mu_row, eps_nu=eps_nu_row, rho=rho_row,
        nitermax=nitermax, stopthr=stopthr
    )
    P_col = cot_col_sparse(
        S, D, A, M, cutoff,
        eps_p=eps_p_col, eps_mu=eps_mu_col, eps_nu=eps_nu_col, rho=rho_col,
        nitermax=nitermax, stopthr=stopthr
    )
    P_blk = cot_blk_sparse(
        S, D, A, M, cutoff,
        eps_p=eps_p_blk, eps_mu=eps_mu_blk, eps_nu=eps_nu_blk, rho=rho_blk,
        nitermax=nitermax, stopthr=stopthr
    )

    P = {}
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if not np.isinf(A[i, j]):
                P[(i, j)] = (
                    float(weights[0]) * P_cot[(i, j)]
                    + float(weights[1]) * P_row[(i, j)]
                    + float(weights[2]) * P_col[(i, j)]
                    + float(weights[3]) * P_blk[(i, j)]
                )
    return P


def cot_sparse(S, D, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, nitermax=1e4, stopthr=1e-8):
    np.set_printoptions(precision=2)
    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape
    max_amount = max(S.sum(), D.sum())
    S = S / max_amount
    D = D / max_amount

    if eps_mu is None:
        eps_mu = eps_p
    if eps_nu is None:
        eps_nu = eps_p
    unot_solver = "momentum" if max(abs(eps_p - eps_mu), abs(eps_p - eps_nu)) > 1e-8 else "sinkhorn"

    a = S.flatten('F')
    b = D.flatten('F')

    C_data, C_row, C_col = [], [], []

    max_cutoff = cutoff.max()
    M_row, M_col = np.where(M <= max_cutoff)
    M_max_sp = sparse.coo_matrix((M[M_row, M_col], (M_row, M_col)), shape=M.shape)

    cost_scales = []
    for i in range(ns_s):
        for j in range(ns_d):
            if not np.isinf(A[i, j]):
                tmp_nzind_s = np.where(S[:, i] > 0)[0]
                tmp_nzind_d = np.where(D[:, j] > 0)[0]
                tmp_M_max_sp = coo_submatrix_pull(M_max_sp, tmp_nzind_s, tmp_nzind_d)
                tmp_ind = np.where(tmp_M_max_sp.data <= cutoff[i, j])[0]
                tmp_row = tmp_nzind_s[tmp_M_max_sp.row[tmp_ind]]
                tmp_col = tmp_nzind_d[tmp_M_max_sp.col[tmp_ind]]
                cost_scales.append(np.max(M_max_sp.data[np.where(M_max_sp.data <= cutoff[i, j])]) * A[i, j])
                C_data.append(tmp_M_max_sp.data[tmp_ind] * A[i, j])
                C_row.append(tmp_row + i * n_pos_s)
                C_col.append(tmp_col + j * n_pos_d)

    cost_scale = np.max(cost_scales)
    C_data = np.concatenate(C_data, axis=0)
    C_row = np.concatenate(C_row, axis=0)
    C_col = np.concatenate(C_col, axis=0)
    C = sparse.coo_matrix((C_data / cost_scale, (C_row, C_col)), shape=(len(a), len(b)))

    nzind_a = np.where(a > 0)[0]
    nzind_b = np.where(b > 0)[0]
    C_nz = coo_submatrix_pull(C, nzind_a, nzind_b)

    del C_data, C_row, C_col, C

    tmp_P = unot(
        a[nzind_a], b[nzind_b], C_nz, eps_p, rho,
        eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=True, solver=unot_solver, nitermax=nitermax, stopthr=stopthr
    )

    del C_nz

    P = sparse.coo_matrix((tmp_P.data, (nzind_a[tmp_P.row], nzind_b[tmp_P.col])), shape=(len(a), len(b)))
    P = P.tocsr()

    P_expand = {}
    for i in range(ns_s):
        for j in range(ns_d):
            if not np.isinf(A[i, j]):
                tmp_P = P[i * n_pos_s:(i + 1) * n_pos_s, j * n_pos_d:(j + 1) * n_pos_d]
                P_expand[(i, j)] = tmp_P.tocoo() * max_amount

    return P_expand


def cot_row_sparse(S, D, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, nitermax=1e4, stopthr=1e-8):
    if eps_mu is None:
        eps_mu = eps_p
    if eps_nu is None:
        eps_nu = eps_p
    unot_solver = "momentum" if max(abs(eps_p - eps_mu), abs(eps_p - eps_nu)) > 1e-8 else "sinkhorn"

    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape

    max_cutoff = cutoff.max()
    M_row, M_col = np.where(M <= max_cutoff)
    M_max_sp = sparse.coo_matrix((M[M_row, M_col], (M_row, M_col)), shape=M.shape)

    P_expand = {}
    for i in range(ns_s):
        a = S[:, i]
        D_ind = np.where(~np.isinf(A[i, :]))[0]
        b = D[:, D_ind].flatten('F')
        nzind_a = np.where(a > 0)[0]
        nzind_b = np.where(b > 0)[0]
        if len(nzind_a) == 0 or len(nzind_b) == 0:
            for j in range(len(D_ind)):
                P_expand[(i, D_ind[j])] = sparse.coo_matrix(([], ([], [])), shape=(n_pos_s, n_pos_d), dtype=float)
            continue
        max_amount = max(a.sum(), b.sum())
        a = a / max_amount
        b = b / max_amount
        C_data, C_row, C_col = [], [], []
        cost_scales = []
        for j in range(len(D_ind)):
            D_j = D_ind[j]
            tmp_nzind_s = np.where(S[:, i] > 0)[0]
            tmp_nzind_d = np.where(D[:, D_j] > 0)[0]
            tmp_M_max_sp = coo_submatrix_pull(M_max_sp, tmp_nzind_s, tmp_nzind_d)
            tmp_ind = np.where(tmp_M_max_sp.data <= cutoff[i, D_j])[0]
            tmp_row = tmp_nzind_s[tmp_M_max_sp.row[tmp_ind]]
            tmp_col = tmp_nzind_d[tmp_M_max_sp.col[tmp_ind]]
            C_data.append(tmp_M_max_sp.data[tmp_ind] * A[i, D_j])
            C_row.append(tmp_row)
            C_col.append(tmp_col + j * n_pos_d)
            cost_scales.append(np.max(M_max_sp.data[np.where(M_max_sp.data <= cutoff[i, D_j])]) * A[i, D_j])
        cost_scale = np.max(cost_scales)
        C_data = np.concatenate(C_data, axis=0)
        C_row = np.concatenate(C_row, axis=0)
        C_col = np.concatenate(C_col, axis=0)
        C = sparse.coo_matrix((C_data / cost_scale, (C_row, C_col)), shape=(len(a), len(b)))

        nzind_a = np.where(a > 0)[0]
        nzind_b = np.where(b > 0)[0]
        C_nz = coo_submatrix_pull(C, nzind_a, nzind_b)

        del C_data, C_row, C_col, C

        tmp_P = unot(
            a[nzind_a], b[nzind_b], C_nz, eps_p, rho,
            eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=True, solver=unot_solver, nitermax=nitermax, stopthr=stopthr
        )

        del C_nz

        P = sparse.coo_matrix((tmp_P.data, (nzind_a[tmp_P.row], nzind_b[tmp_P.col])), shape=(len(a), len(b)))
        P = P.tocsr()

        for j in range(len(D_ind)):
            tmp_P = P[:, j * n_pos_d:(j + 1) * n_pos_d]
            P_expand[(i, D_ind[j])] = tmp_P.tocoo() * max_amount

        del P

    return P_expand


def cot_col_sparse(S, D, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, nitermax=1e4, stopthr=1e-8):
    if eps_mu is None:
        eps_mu = eps_p
    if eps_nu is None:
        eps_nu = eps_p
    unot_solver = "momentum" if max(abs(eps_p - eps_mu), abs(eps_p - eps_nu)) > 1e-8 else "sinkhorn"

    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape

    max_cutoff = cutoff.max()
    M_row, M_col = np.where(M <= max_cutoff)
    M_max_sp = sparse.coo_matrix((M[M_row, M_col], (M_row, M_col)), shape=M.shape)

    P_expand = {}
    for j in range(ns_d):
        S_ind = np.where(~np.isinf(A[:, j]))[0]
        a = S[:, S_ind].flatten('F')
        b = D[:, j]
        nzind_a = np.where(a > 0)[0]
        nzind_b = np.where(b > 0)[0]
        if len(nzind_a) == 0 or len(nzind_b) == 0:
            for i in range(len(S_ind)):
                P_expand[(S_ind[i], j)] = sparse.coo_matrix(([], ([], [])), shape=(n_pos_s, n_pos_d), dtype=float)
            continue
        max_amount = max(a.sum(), b.sum())
        a = a / max_amount
        b = b / max_amount
        C_data, C_row, C_col = [], [], []
        cost_scales = []
        for i in range(len(S_ind)):
            S_i = S_ind[i]
            tmp_nzind_s = np.where(S[:, S_i] > 0)[0]
            tmp_nzind_d = np.where(D[:, j] > 0)[0]
            tmp_M_max_sp = coo_submatrix_pull(M_max_sp, tmp_nzind_s, tmp_nzind_d)
            tmp_ind = np.where(tmp_M_max_sp.data <= cutoff[S_i, j])[0]
            tmp_row = tmp_nzind_s[tmp_M_max_sp.row[tmp_ind]]
            tmp_col = tmp_nzind_d[tmp_M_max_sp.col[tmp_ind]]
            C_data.append(tmp_M_max_sp.data[tmp_ind] * A[S_i, j])
            C_row.append(tmp_row + i * n_pos_s)
            C_col.append(tmp_col)
            cost_scales.append(np.max(M_max_sp.data[np.where(M_max_sp.data <= cutoff[S_i, j])]) * A[S_i, j])
        cost_scale = np.max(cost_scales)
        C_data = np.concatenate(C_data, axis=0)
        C_row = np.concatenate(C_row, axis=0)
        C_col = np.concatenate(C_col, axis=0)
        C = sparse.coo_matrix((C_data / cost_scale, (C_row, C_col)), shape=(len(a), len(b)))

        nzind_a = np.where(a > 0)[0]
        nzind_b = np.where(b > 0)[0]
        C_nz = coo_submatrix_pull(C, nzind_a, nzind_b)

        del C_data, C_row, C_col, C

        tmp_P = unot(
            a[nzind_a], b[nzind_b], C_nz, eps_p, rho,
            eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=True, solver=unot_solver, nitermax=nitermax, stopthr=stopthr
        )

        del C_nz

        P = sparse.coo_matrix((tmp_P.data, (nzind_a[tmp_P.row], nzind_b[tmp_P.col])), shape=(len(a), len(b)))
        P = P.tocsr()

        for i in range(len(S_ind)):
            tmp_P = P[i * n_pos_s:(i + 1) * n_pos_s, :]
            P_expand[(S_ind[i], j)] = tmp_P.tocoo() * max_amount

        del P

    return P_expand


def cot_blk_sparse(S, D, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, nitermax=1e4, stopthr=1e-8):
    if eps_mu is None:
        eps_mu = eps_p
    if eps_nu is None:
        eps_nu = eps_p
    unot_solver = "momentum" if max(abs(eps_p - eps_mu), abs(eps_p - eps_nu)) > 1e-8 else "sinkhorn"

    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape

    max_cutoff = cutoff.max()
    M_row, M_col = np.where(M <= max_cutoff)
    M_max_sp = sparse.coo_matrix((M[M_row, M_col], (M_row, M_col)), shape=M.shape)

    P_expand = {}
    for i in range(ns_s):
        for j in range(ns_d):
            if not np.isinf(A[i, j]):
                a = S[:, i]
                b = D[:, j]
                nzind_a = np.where(a > 0)[0]
                nzind_b = np.where(b > 0)[0]
                if len(nzind_a) == 0 or len(nzind_b) == 0:
                    P_expand[(i, j)] = sparse.coo_matrix(([], ([], [])), shape=(n_pos_s, n_pos_d), dtype=float)
                    continue
                max_amount = max(a.sum(), b.sum())
                a = a / max_amount
                b = b / max_amount
                tmp_nzind_s = np.where(S[:, i] > 0)[0]
                tmp_nzind_d = np.where(D[:, j] > 0)[0]
                tmp_M_max_sp = coo_submatrix_pull(M_max_sp, tmp_nzind_s, tmp_nzind_d)
                tmp_ind = np.where(tmp_M_max_sp.data <= cutoff[i, j])[0]
                tmp_row = tmp_nzind_s[tmp_M_max_sp.row[tmp_ind]]
                tmp_col = tmp_nzind_d[tmp_M_max_sp.col[tmp_ind]]
                C_data = tmp_M_max_sp.data[tmp_ind] * A[i, j]
                cost_scale = np.max(M_max_sp.data[np.where(M_max_sp.data <= cutoff[i, j])]) * A[i, j]
                C = sparse.coo_matrix((C_data / cost_scale, (tmp_row, tmp_col)), shape=(len(a), len(b)))

                nzind_a = np.where(a > 0)[0]
                nzind_b = np.where(b > 0)[0]
                C_nz = coo_submatrix_pull(C, nzind_a, nzind_b)

                del C_data, C

                tmp_P = unot(
                    a[nzind_a], b[nzind_b], C_nz, eps_p, rho,
                    eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=True, solver=unot_solver, nitermax=nitermax, stopthr=stopthr
                )

                del C_nz

                P = sparse.coo_matrix((tmp_P.data, (nzind_a[tmp_P.row], nzind_b[tmp_P.col])), shape=(len(a), len(b)))
                P_expand[(i, j)] = P * max_amount

    return P_expand


def coo_submatrix_pull(matr, rows, cols):
    if type(matr) != sparse.coo_matrix:
        raise TypeError('Matrix must be sparse COOrdinate format')

    gr = -1 * np.ones(matr.shape[0])
    gc = -1 * np.ones(matr.shape[1])

    lr = len(rows)
    lc = len(cols)

    ar = np.arange(0, lr)
    ac = np.arange(0, lc)
    gr[rows[ar]] = ar
    gc[cols[ac]] = ac
    mrow = matr.row
    mcol = matr.col
    newelem = (gr[mrow] > -1) & (gc[mcol] > -1)
    newrows = mrow[newelem]
    newcols = mcol[newelem]
    return sparse.coo_matrix((matr.data[newelem], np.array([gr[newrows], gc[newcols]])), (lr, lc))


def unot(a, b, C, eps_p, rho, eps_mu=None, eps_nu=None, sparse_mtx=False, solver="sinkhorn", nitermax=10000, stopthr=1e-8, momentum_dt=0.1, momentum_beta=0.0):
    if eps_mu is None:
        eps_mu = eps_p
    if eps_nu is None:
        eps_nu = eps_p
    nzind_a = np.where(a > 0)[0]
    nzind_b = np.where(b > 0)[0]
    if len(nzind_a) == 0 or len(nzind_b) == 0:
        if sparse_mtx:
            return sparse.coo_matrix(([], ([], [])), shape=(len(a), len(b)))
        return np.zeros([len(a), len(b)], float)
    if solver == "sinkhorn" and max(abs(eps_p - eps_mu), abs(eps_p - eps_nu)) > 1e-8:
        raise ValueError("To use Sinkhorn algorithm, set eps_p=eps_mu=eps_nu")
    if solver == "sinkhorn" and not sparse_mtx:
        return unot_sinkhorn_l1_dense(a, b, C, eps_p, rho, nitermax=nitermax, stopthr=stopthr)
    if solver == "sinkhorn" and sparse_mtx:
        return unot_sinkhorn_l1_sparse(a, b, C, eps_p, rho, nitermax=nitermax, stopthr=stopthr)
    if solver == "momentum" and not sparse_mtx:
        return unot_momentum_l1_dense(
            a, b, C, eps_p, eps_mu, eps_nu, rho,
            nitermax=nitermax, stopthr=stopthr, dt=momentum_dt, beta=momentum_beta, precondition=True
        )
    raise NotImplementedError("Sparse momentum solver is not implemented.")


def unot_sinkhorn_l1_dense(a, b, C, eps, m, nitermax=10000, stopthr=1e-8, output_fg=False):
    f = np.zeros_like(a)
    g = np.zeros_like(b)
    niter = 0
    err = 100
    while niter <= nitermax and err > stopthr:
        fprev = f
        gprev = g
        f = (
            eps * np.log(a)
            - eps * np.log(np.sum(np.exp((f.reshape(-1, 1) + g.reshape(1, -1) - C) / eps), axis=1) + np.exp((-m + f) / eps))
            + f
        )
        g = (
            eps * np.log(b)
            - eps * np.log(np.sum(np.exp((f.reshape(-1, 1) + g.reshape(1, -1) - C) / eps), axis=0) + np.exp((-m + g) / eps))
            + g
        )
        if niter % 10 == 0:
            err_f = abs(f - fprev).max() / max(abs(f).max(), abs(fprev).max(), 1.)
            err_g = abs(g - gprev).max() / max(abs(g).max(), abs(gprev).max(), 1.)
            err = 0.5 * (err_f + err_g)
        niter += 1
    if output_fg:
        return f, g
    return np.exp((f.reshape(-1, 1) + g.reshape(1, -1) - C) / eps)


def unot_sinkhorn_l1_sparse(a, b, C, eps, m, nitermax=10000, stopthr=1e-8):
    tmp_K = C.copy()
    f = np.zeros_like(a)
    g = np.zeros_like(b)
    niter = 0
    err = 100
    while niter <= nitermax and err > stopthr:
        fprev = f
        gprev = g
        tmp_K.data = np.exp((-C.data + f[C.row] + g[C.col]) / eps)
        f = eps * np.log(a) - eps * np.log(np.sum(tmp_K, axis=1).A.reshape(-1) + np.exp((-m + f) / eps)) + f
        tmp_K.data = np.exp((-C.data + f[C.row] + g[C.col]) / eps)
        g = eps * np.log(b) - eps * np.log(np.sum(tmp_K, axis=0).A.reshape(-1) + np.exp((-m + g) / eps)) + g
        if niter % 10 == 0:
            err_f = abs(f - fprev).max() / max(abs(f).max(), abs(fprev).max(), 1.)
            err_g = abs(g - gprev).max() / max(abs(g).max(), abs(gprev).max(), 1.)
            err = 0.5 * (err_f + err_g)
        niter += 1
    tmp_K.data = np.exp((-C.data + f[C.row] + g[C.col]) / eps)
    return tmp_K


def unot_momentum_l1_dense(a, b, C, eps_p, eps_mu, eps_nu, m, nitermax=1e4, stopthr=1e-8, dt=0.01, beta=0.8, precondition=False):
    f = np.zeros_like(a)
    g = np.zeros_like(b)
    if precondition:
        f, g = unot_sinkhorn_l1_dense(a, b, C, eps_p, m, output_fg=True)
    f_old = np.array(f)
    g_old = np.array(g)
    F_old = np.array(f)
    G_old = np.array(g)
    niter = 0
    err = 100

    def Qf(ff, gg, ee_p, ee_mu, mm, aa, CC):
        return np.exp(ff / ee_p) * np.sum(np.exp((gg.reshape(1, -1) - CC) / ee_p), axis=1) + np.exp((ff - mm) / ee_mu) - aa

    def Qg(ff, gg, ee_p, ee_nu, mm, bb, CC):
        return np.exp(gg / ee_p) * np.sum(np.exp((ff.reshape(-1, 1) - CC) / ee_p), axis=0) + np.exp((gg - mm) / ee_nu) - bb

    while niter <= nitermax and err > stopthr:
        F = beta * F_old + Qf(f_old, g_old, eps_p, eps_mu, m, a, C)
        f = f_old - dt * F
        G = beta * G_old + Qg(f_old, g_old, eps_p, eps_nu, m, b, C)
        g = g_old - dt * G
        if niter % 10 == 0:
            err_f = abs(f - f_old).max() / max(abs(f).max(), abs(f_old).max(), 1.)
            err_g = abs(g - g_old).max() / max(abs(g).max(), abs(g_old).max(), 1.)
            err = 0.5 * (err_f + err_g)
        f_old[:] = f[:]
        F_old[:] = F[:]
        g_old[:] = g[:]
        G_old[:] = G[:]
        niter += 1
    return np.exp((f.reshape(-1, 1) + g.reshape(1, -1) - C) / eps_p)
