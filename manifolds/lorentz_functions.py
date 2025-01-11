import torch


def Lorentzian_vector(x, gamma=1.0):
    x_0 = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + gamma)
    x = torch.cat([x_0, x], dim=-1)
    return x


def Lorentzian_inner_product(x, y, keepdim=False):
    # Utilize the broadcast mechanism of PyTorch
    xy = x * y
    if xy.dim() == 3:
        xy[:, :, 0] = - xy[:, :, 0]
    elif xy.dim() == 2:
        xy[:, 0] = - xy[:, 0]
    else:
        raise ValueError("tensor dim should be 2 or 3.")
    L_product = torch.sum(xy, dim=-1, keepdim=keepdim)
    return L_product


def squared_lorentzian_distance(x, y, gamma=1.0, hyper_input=True, keepdim=False):
    if hyper_input is False:
        x = Lorentzian_vector(x, gamma=gamma)
        y = Lorentzian_vector(y, gamma=gamma)
    xy = Lorentzian_inner_product(x=x, y=y, keepdim=keepdim)
    sL_distance = -2 * gamma - 2 * xy
    return sL_distance


def Centroid(x, y, gamma, weights=None):
    if isinstance(gamma, torch.Tensor) is False:
        gamma = torch.tensor(gamma)
    if weights is not None:
        a = weights[0] * x + weights[1] * y
    else:
        a = x + y
    modulus_a = torch.sqrt(Lorentzian_inner_product(a, a, keepdim=True).abs())
    mu = torch.sqrt(gamma) * a / modulus_a
    return mu


def Centroid_group(x, gamma=1.0, weights=None):
    if isinstance(gamma, torch.Tensor) is False:
        gamma = torch.tensor(gamma)
    if weights is not None:
        if isinstance(weights, torch.Tensor) is False:
            weights = torch.tensor(weights).unsqueeze(1).to(x.device)
        x = x * weights
        a = torch.sum(x, dim=1)
    else:
        a = torch.sum(x, dim=1)
    modulus_a = torch.sqrt(Lorentzian_inner_product(a, a, keepdim=True).abs())
    mu = torch.sqrt(gamma) * a / modulus_a
    return mu


def Centroid_aggregation(x_list, gamma=1.0, weights=None):
    B, L, d = x_list[0].size()
    for i in range(len(x_list)):
        x_list[i] = x_list[i].reshape(-1, 1, d)
    x_list = torch.cat(x_list, dim=1)
    aggre_emb = Centroid_group(x_list, gamma=gamma, weights=weights)

    return aggre_emb.reshape(B, L, -1)


def Centroid_group_euc(x, gamma=1.0, weights=None):
    x = Lorentzian_vector(x, gamma=gamma)
    mu = Centroid_group(x, gamma=1.0, weights=weights)
    return mu


def givens_rotations(r, x):
    """ Rotation operation. """
    givens = r.view((r.shape[0], -1, 2))
    # givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True)
    # with clamp_min ?
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    x = x.view((r.shape[0], -1, 2))
    # first: cos, second: sin
    x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)

    return x_rot.view((r.shape[0], -1))


def distance_prediction_scores(x, y, c=1.0, is_neg=True):
    x, y = x.unsqueeze(2), y.unsqueeze(1)
    x2_srt = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + c)
    y2_srt = -torch.sqrt(torch.sum(y * y, dim=-1, keepdim=True) + c)
    u = torch.cat((x, x2_srt), -1)
    v = torch.cat((y, y2_srt), -1)
    vt = v.squeeze(1).T
    uv = u @ vt
    result = - 2 * c - 2 * uv
    if is_neg:
        result = result.neg()

    return result.squeeze(2)


if __name__ == "__main__":
    gamma = 1.0
    emb = torch.randn(3, 5)
    emb = Lorentzian_vector(emb, gamma=gamma)
    dis = squared_lorentzian_distance(emb, emb, gamma=gamma)
    print(dis)
