# import geoopt
# import torch

# stiefel = geoopt.manifolds.Stiefel()
# torch.manual_seed(0)
# initial_point = torch.randn(p, d)
# initial_point, _ = torch.qr(initial_point)

# v = geoopt.ManifoldParameter(initial_point, manifold=stiefel)
# print(v)

# optimizer = RiemannianSGD([v], lr=0.1)
# #  nesterov=False, momentum=0.5, dampening=0.1

# max_iterations = 2

# for step in range(max_iterations):
#     vt = v.clone(
#     optimizer.zero_grad()
#     cost = f(A, B, a, v, At, Bt)
#     gradient = grad(A, B, a, v, At, Bt).to(torch.float)
#     v.grad = gradient
#     optimizer.step()
#     vt_plus = v.clone()

#     if stepExit(vt_plus, vt, cost, A, B, At, Bt, a):
#         break

# print(step)
# print(v)


# def f(A, B, a, v, At, Bt):
#     # v = v.data
#     # v = v.to(torch.double)

#     bv_inv = torch.inverse(v.t() @ B @ v)
#     va = v.t() @ A @ v
#     bv_t_inv = torch.inverse(v.t() @ Bt @ v)
#     va_t = v.t() @ At @ v

#     f_v = -torch.trace(va @ bv_inv) + a * torch.trace(va_t @ bv_t_inv)
#     return f_v


# def grad(A, B, a, v, At, Bt):
#     # v = v.data
#     # v = v.to(torch.double)

#     bv_inv = torch.inverse(v.t() @ B @ v)
#     va = v.t() @ A @ v
#     bv_t_inv = torch.inverse(v.t() @ Bt @ v)
#     va_t = v.t() @ At @ v

#     avb = A @ v @ bv_inv - B @ v @ bv_inv @ va @ bv_inv
#     avb_t = At @ v @ bv_t_inv - Bt @ v @ bv_t_inv @ va_t @ bv_t_inv

#     gradient = -2 * (avb - a * avb_t)
#     return gradient


# def f(A, B, a, v, At, Bt):
#     # v = v.data
#     # v = v.to(torch.double)
#     bv_inv = np.linalg.inv(np.transpose(v) @ B @ v)
#     va = np.transpose(v) @ A @ v
#     bv_t_inv = np.linalg.inv(np.transpose(v) @ Bt @ v)
#     va_t = np.transpose(v) @ At @ v
#     f_v = -np.matrix.trace(va @ bv_inv) + a * np.matrix.trace(va_t @ bv_t_inv)
#     # bv_inv = torch.inverse(v.t() @ B @ v)

#     # bv_t_inv = torch.inverse(v.t() @ Bt @ v)
#     # f_v = -torch.trace(va @ bv_inv) + a * torch.trace(va_t @ bv_t_inv)
#     return f_v


# def grad(A, B, a, v, At, Bt):
#     # v = v.data
#     # v = v.to(torch.double)
#     bv_inv = np.linalg.inv(np.transpose(v) @ B @ v)
#     va = np.transpose(v) @ A @ v
#     bv_t_inv = np.linalg.inv(np.transpose(v) @ Bt @ v)
#     va_t = np.transpose(v) @ At @ v
#     # bv_inv = torch.inverse(v.t() @ B @ v)
#     # va = v.t() @ A @ v
#     # bv_t_inv = torch.inverse(v.t() @ Bt @ v)
#     # va_t = v.t() @ At @ v

#     avb = A @ v @ bv_inv - B @ v @ bv_inv @ va @ bv_inv
#     avb_t = At @ v @ bv_t_inv - Bt @ v @ bv_t_inv @ va_t @ bv_t_inv

#     gradient = -2 * (avb - a * avb_t)
#     return gradient


#   matrix = np.random.normal(size=(p, d))
#     manifold_ = Stiefel(p, d)
#     cost_ = f(A, B, a, matrix, At, Bt)
#     gradient = grad(A, B, a, matrix, At, Bt)
#     problem = pymanopt.Problem(
#         manifold=manifold_, cost=cost_, euclidean_gradient=gradient)
#     optimizer = steepest_descent()
#     result = optimizer.run(problem)

#     print(result.point)
