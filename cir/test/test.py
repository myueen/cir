# # import geoopt
# # import torch

# ## Dependencies
# This project relies on the following external packages:
# - [sliced](github.com/joshloyal/sliced) by Joshya Loyal


# # Center the matrix X by subtracting the original matrix X by the column means of X
#     X = X - np.mean(X, axis=0)

#     # Covariance matrix
#     cov_X = X.cov()

#     # Define H, which represents the # of intervals I that splits range(Y)
#     Y_unique_value = Y.nunique().item()
#     if Y_unique_value == 2:
#         H = 2                       # number of slices
#     elif 2 < Y_unique_value <= 10:
#         H = Y_unique_value
#     else:
#         if d <= 2:
#             H = 10
#         else:
#             H = 4

#     # Count the of ocurrence of y in each H interval
#     unique_y_vals_F, counts_F = unique_counts(Y)
#     cumsum_y_F = np.cumsum(counts_F)
#     n_y_values_F = unique_y_vals_F.shape[0]

#     if H >= n_y_values_F:
#         if H > n_y_values_F:
#             warnings.warn(
#                 "n_slices greater than the number of unique y values.")

#         slice_partition_F = np.hstack((0, cumsum_y_F))
#     else:
#         n_obs_F = np.floor(Y.shape[0] / H)
#         n_samples_seen_F = 0
#         slice_partition_F = [0]  # index in y of start of a new slice
#         while n_samples_seen_F < Y.shape[0] - 2:
#             slice_start_F = np.where(
#                 cumsum_y_F >= n_samples_seen_F + n_obs_F)[0]
#             if slice_start_F.shape[0] == 0:  # this means we've reached the end
#                 slice_start_F = cumsum_y_F.shape[0] - 1
#             else:
#                 slice_start_F = slice_start_F[0]

#             n_samples_seen_F = cumsum_y_F[slice_start_F]
#             slice_partition_F.append(n_samples_seen_F)

#     slice_indicator_F = np.ones(Y.shape[0], dtype=int)

#     for j, (start_idx, end_idx) in enumerate(zip(slice_partition_F, slice_partition_F[1:])):
#         if j == len(slice_partition_F) - 2:
#             slice_indicator_F[start_idx:] = j
#         else:
#             slice_indicator_F[start_idx:end_idx] = j

#     slice_counts_F = np.bincount(slice_indicator_F)

#     # Define Sigma X
#     Q, R = qr(X, mode='economic')
#     Z = np.sqrt(n) * Q
#     Y = Y.to_numpy()
#     sorted_indices = np.argsort(Y, axis=0, kind='stable')
#     Z = Z[sorted_indices, :]
#     Z = np.squeeze(Z)

#     M_means = grouped_sum(Z, slice_indicator_F) / \
#         np.sqrt(slice_counts_F.reshape(-1, 1))

#     sigma_X = np.dot(M_means.T, M_means) / n

#     eigenvalues, eigenvectors = np.linalg.eigh(sigma_X)
#     epsilon = 2 * abs(np.min(eigenvalues))
#     sigma_X = sigma_X + epsilon * np.identity(p)

#     cov_X = np.array(cov_X)

#     # Generalized Eigenvalue Decomposition
#     A = cov_X @ sigma_X @ cov_X
#     B = cov_X @ cov_X
#     eigenvalues, eigenvectors = eigh(cov_X, sigma_X)

#     if alpha == 0:
#         v = eigenvectors[:, :d]
#         f_v = -1 * (np.trace(v.T @ A @ v @ np.linalg.inv(v.T @ B @ v)))
#         return v, f_v

#     # --------The following is for background data and the caase when a > 0-------
#     # Center the data
#     Xt = Xt - np.mean(Xt, axis=0)

#     # Covariance matrix
#     cov_Xt = Xt.cov()

#     # Define Ht, which represents the # of interval I that splits range(Yt)
#     Yt_unique_value = Yt.nunique().item()
#     if Yt_unique_value == 2:
#         Ht = 2
#     elif 2 < Yt_unique_value <= 10:
#         Ht = Yt_unique_value
#     else:
#         if d <= 2:
#             Ht = 10
#         else:
#             Ht = 4

#     # Count the of ocurrence of y in each H intervaL
#     unique_y_vals_B, counts_B = unique_counts(Yt)
#     cumsum_y_B = np.cumsum(counts_B)
#     n_y_values_B = unique_y_vals_B.shape[0]

#     if Ht >= n_y_values_B:
#         if Ht > n_y_values_B:
#             warnings.warn(
#                 "n_slices greater than the number of unique y values.")

#         slice_partition_B = np.hstack((0, cumsum_y_B))
#     else:
#         n_obs_B = np.floor(Yt.shape[0] / Ht)
#         n_samples_seen_B = 0
#         slice_partition_B = [0]  # index in y of start of a new slice
#         while n_samples_seen_B < Yt.shape[0] - 2:
#             slice_start_B = np.where(
#                 cumsum_y_B >= n_samples_seen_B + n_obs_B)[0]
#             if slice_start_B.shape[0] == 0:  # this means we've reached the end
#                 slice_start_B = cumsum_y_B.shape[0] - 1
#             else:
#                 slice_start_B = slice_start_B[0]

#             n_samples_seen_B = cumsum_y_B[slice_start_B]
#             slice_partition_B.append(n_samples_seen_B)

#     slice_indicator_B = np.ones(Yt.shape[0], dtype=int)
#     for j, (start_idx, end_idx) in enumerate(zip(slice_partition_B, slice_partition_B[1:])):
#         if j == len(slice_partition_B) - 2:
#             slice_indicator_B[start_idx:] = j
#         else:
#             slice_indicator_B[start_idx:end_idx] = j

#     slice_counts_B = np.bincount(slice_indicator_B)

#     # Define Sigma X
#     Qt, Rt = qr(Xt, mode='economic')
#     Zt = np.sqrt(m) * Qt
#     Yt = Yt.to_numpy()
#     sorted_indices_t = np.argsort(Yt, axis=0, kind='stable')
#     Zt = Zt[sorted_indices_t, :]
#     Zt = np.squeeze(Zt)
#     Mt_means = grouped_sum(Zt, slice_indicator_B) / \
#         np.sqrt(slice_counts_B.reshape(-1, 1))

#     sigma_Xt = np.dot(Mt_means.T, Mt_means) / m

#     cov_Xt = np.array(cov_Xt)


# # stiefel = geoopt.manifolds.Stiefel()
# # torch.manual_seed(0)
# # initial_point = torch.randn(p, d)
# # initial_point, _ = torch.qr(initial_point)

# # v = geoopt.ManifoldParameter(initial_point, manifold=stiefel)
# # print(v)

# # optimizer = RiemannianSGD([v], lr=0.1)
# # #  nesterov=False, momentum=0.5, dampening=0.1

# # max_iterations = 2

# # for step in range(max_iterations):
# #     vt = v.clone(
# #     optimizer.zero_grad()
# #     cost = f(A, B, a, v, At, Bt)
# #     gradient = grad(A, B, a, v, At, Bt).to(torch.float)
# #     v.grad = gradient
# #     optimizer.step()
# #     vt_plus = v.clone()

# #     if stepExit(vt_plus, vt, cost, A, B, At, Bt, a):
# #         break

# # print(step)
# # print(v)


# # def f(A, B, a, v, At, Bt):
# #     # v = v.data
# #     # v = v.to(torch.double)

# #     bv_inv = torch.inverse(v.t() @ B @ v)
# #     va = v.t() @ A @ v
# #     bv_t_inv = torch.inverse(v.t() @ Bt @ v)
# #     va_t = v.t() @ At @ v

# #     f_v = -torch.trace(va @ bv_inv) + a * torch.trace(va_t @ bv_t_inv)
# #     return f_v


# # def grad(A, B, a, v, At, Bt):
# #     # v = v.data
# #     # v = v.to(torch.double)

# #     bv_inv = torch.inverse(v.t() @ B @ v)
# #     va = v.t() @ A @ v
# #     bv_t_inv = torch.inverse(v.t() @ Bt @ v)
# #     va_t = v.t() @ At @ v

# #     avb = A @ v @ bv_inv - B @ v @ bv_inv @ va @ bv_inv
# #     avb_t = At @ v @ bv_t_inv - Bt @ v @ bv_t_inv @ va_t @ bv_t_inv

# #     gradient = -2 * (avb - a * avb_t)
# #     return gradient


# # def f(A, B, a, v, At, Bt):
# #     # v = v.data
# #     # v = v.to(torch.double)
# #     bv_inv = np.linalg.inv(np.transpose(v) @ B @ v)
# #     va = np.transpose(v) @ A @ v
# #     bv_t_inv = np.linalg.inv(np.transpose(v) @ Bt @ v)
# #     va_t = np.transpose(v) @ At @ v
# #     f_v = -np.matrix.trace(va @ bv_inv) + a * np.matrix.trace(va_t @ bv_t_inv)
# #     # bv_inv = torch.inverse(v.t() @ B @ v)

# #     # bv_t_inv = torch.inverse(v.t() @ Bt @ v)
# #     # f_v = -torch.trace(va @ bv_inv) + a * torch.trace(va_t @ bv_t_inv)
# #     return f_v


# # def grad(A, B, a, v, At, Bt):
# #     # v = v.data
# #     # v = v.to(torch.double)
# #     bv_inv = np.linalg.inv(np.transpose(v) @ B @ v)
# #     va = np.transpose(v) @ A @ v
# #     bv_t_inv = np.linalg.inv(np.transpose(v) @ Bt @ v)
# #     va_t = np.transpose(v) @ At @ v
# #     # bv_inv = torch.inverse(v.t() @ B @ v)
# #     # va = v.t() @ A @ v
# #     # bv_t_inv = torch.inverse(v.t() @ Bt @ v)
# #     # va_t = v.t() @ At @ v

# #     avb = A @ v @ bv_inv - B @ v @ bv_inv @ va @ bv_inv
# #     avb_t = At @ v @ bv_t_inv - Bt @ v @ bv_t_inv @ va_t @ bv_t_inv

# #     gradient = -2 * (avb - a * avb_t)
# #     return gradient


# #   matrix = np.random.normal(size=(p, d))
# #     manifold_ = Stiefel(p, d)
# #     cost_ = f(A, B, a, matrix, At, Bt)
# #     gradient = grad(A, B, a, matrix, At, Bt)
# #     problem = pymanopt.Problem(
# #         manifold=manifold_, cost=cost_, euclidean_gradient=gradient)
# #     optimizer = steepest_descent()
# #     result = optimizer.run(problem)

# #     print(result.point)
