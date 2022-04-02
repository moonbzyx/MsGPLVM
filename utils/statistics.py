import torch
# import numpy as np
# import matplotlib.pyplot as plt


class Psi():
    def __init__(self, phi, sigma, w, mu, S, Xu, device):

        self.phi = phi  # N * D * L
        self.sigma = sigma**2  # L
        self.w = w  # L *Q
        self.mu = mu  # Q * N
        self.S = S  # Q * N
        self.N, self.D, self.L = self.phi.shape
        self.Xu = Xu  # L*M*Q
        self.Q = w.shape[1]
        self.M = Xu.shape[1]
        self.device = device
        # self.set_device()

        # print('L', self.L, 'D', self.D, 'N', self.N, 'M', self.M, 'Q', self.Q)

    # def set_device(self):
    #     self.device = torch.device('cuda')
    #     self.M.to(self.device)

    def psi_0(self):
        phi = self.phi.permute([2, 1, 0]).sum(dim=-1)
        variance = self.sigma.unsqueeze(-1).expand([self.L, self.D])
        psi = torch.einsum('ij,ij->ij', [phi, variance])
        return psi

    def psi_1(self):
        mu = self.mu.permute([1, 0]).unsqueeze(1).expand(
            ([self.L, self.D, self.N, self.M, self.Q]))
        z = self.Xu.unsqueeze(1).unsqueeze(1).expand(
            ([self.L, self.D, self.N, self.M, self.Q]))
        alpha = self.w.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(
            ([self.L, self.D, self.N, self.M, self.Q]))
        S = self.S.permute([1, 0]).unsqueeze(1).expand(
            ([self.L, self.D, self.N, self.M, self.Q]))
        sigma = self.sigma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
            ([self.L, self.D, self.N, self.M]))
        phi = self.phi.permute([2, 1, 0]).unsqueeze(-1).expand(
            ([self.L, self.D, self.N, self.M]))

        dist = (mu - z)**2
        sub_denominator = alpha * S + 1.0
        coe_1 = -0.5 * alpha / sub_denominator
        part_1 = torch.exp((dist * coe_1).sum(dim=-1))
        part_2 = torch.prod(1.0 / torch.sqrt(sub_denominator), dim=-1)
        psi = phi * sigma * part_1 * part_2
        return psi.to(self.device)
        # print(psi.shape)

    def psi_11(self):
        # self.phi = phi  # N * D * L
        # self.sigma = sigma**2  # L
        # self.w = w  # L *Q
        # self.mu = mu  # Q * N
        # self.S = S  # Q * N
        # self.Xu = Xu # L * M * Q
        psi = []
        for l in range(self.L):
            psi_d = []
            for d in range(self.D):
                mu = self.mu.permute([1, 0]).unsqueeze(1).expand(
                    ([self.N, self.M, self.Q]))
                z = self.Xu[l].expand(([self.N, self.M, self.Q]))
                alpha = self.w[l].expand(([self.N, self.M, self.Q]))
                S = self.S.permute([1, 0]).unsqueeze(1).expand(
                    ([self.N, self.M, self.Q]))
                sigma = self.sigma[l]
                phi = self.phi[:, d, l].unsqueeze(-1).expand(([self.N,
                                                               self.M]))

                dist = (mu - z)**2
                sub_denominator = alpha * S + 1.0
                coe_1 = -0.5 * alpha / sub_denominator
                part_1 = torch.exp((dist * coe_1).sum(dim=-1))
                part_2 = torch.prod(1.0 / torch.sqrt(sub_denominator), dim=-1)
                psi_nm = phi * sigma * part_1 * part_2
                psi_d.append(psi_nm)
            psi_d = torch.stack(psi_d, dim=0)
            psi.append(psi_d)
        psi = torch.stack(psi, dim=0)
        return psi

    def psi_1_cpu(self):
        mu = self.mu.permute([1, 0]).unsqueeze(1).expand(
            ([self.L, self.D, self.N, self.M, self.Q])).cpu()
        z = self.Xu.unsqueeze(1).unsqueeze(1).expand(
            ([self.L, self.D, self.N, self.M, self.Q])).cpu()
        alpha = self.w.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(
            ([self.L, self.D, self.N, self.M, self.Q])).cpu()
        S = torch.stack([torch.diag(self.S[i]) for i in range(self.Q)
                         ]).permute([1, 0]).unsqueeze(1).expand(
                             ([self.L, self.D, self.N, self.M, self.Q])).cpu()
        sigma = self.sigma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
            ([self.L, self.D, self.N, self.M])).cpu()
        phi = self.phi.permute([2, 1, 0]).unsqueeze(-1).expand(
            ([self.L, self.D, self.N, self.M])).cpu()

        dist = (mu - z)**2
        sub_denominator = alpha * S + 1.0
        coe_1 = -0.5 * alpha / sub_denominator
        part_1 = torch.exp((dist * coe_1).sum(dim=-1))
        part_2 = torch.prod(1.0 / torch.sqrt(sub_denominator), dim=-1)
        psi = phi * sigma * part_1 * part_2
        return psi.to(self.device)
        # print(psi.shape)

    def psi_2(self):
        mu = self.mu.permute([1, 0]).unsqueeze(1).unsqueeze(1).expand(
            ([self.L, self.D, self.N, self.M, self.M, self.Q]))
        z = self.Xu.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(
            ([self.L, self.D, self.N, self.M, self.M, self.Q]))
        z_t = self.Xu.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(
            ([self.L, self.D, self.N, self.M, self.M,
              self.Q])).permute([0, 1, 2, 4, 3, 5])
        alpha = self.w.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(
            1).expand(([self.L, self.D, self.N, self.M, self.M, self.Q]))
        S = self.S.permute([1, 0]).unsqueeze(1).unsqueeze(1).expand(
            ([self.L, self.D, self.N, self.M, self.M, self.Q]))
        sigma = (self.sigma**2).unsqueeze(-1).unsqueeze(-1).unsqueeze(
            -1).unsqueeze(-1).expand(([self.L, self.D, self.N, self.M,
                                       self.M]))
        phi = self.phi.permute([2, 1, 0]).unsqueeze(-1).unsqueeze(-1).expand(
            ([self.L, self.D, self.N, self.M, self.M]))

        z_m = (z + z_t) / 2.0
        dist_1 = (z - z_t)**2.0
        dist_2 = (mu - z_m)**2.0
        sub_denominator = 2.0 * alpha * S + 1

        coe_1 = -0.25 * alpha
        coe_2 = -alpha / sub_denominator

        part_1 = torch.exp((dist_1 * coe_1 + dist_2 * coe_2).sum(dim=-1))
        part_2 = torch.prod(1.0 / torch.sqrt(sub_denominator), dim=-1)
        psi = (phi * sigma * part_1 * part_2).sum(dim=2)
        return psi

        # print(psi.shape)
        # print(coe_2[0, 0, 0, 2, 2, 0])
        # print(alpha[0, 0, 0, 2, 2, 0])
        # print(z_t[0, 0, 0, 2, 2, 0])

    def psi_22(self):
        # self.phi = phi  # N * D * L
        # self.sigma = sigma**2  # L
        # self.w = w  # L *Q
        # self.mu = mu  # Q * N
        # self.S = S  # Q * N
        # self.Xu = Xu # L * M * Q

        psi = []
        for l in range(self.L):
            psi_d = []
            for d in range(self.D):
                psi_mm = torch.zeros([self.M, self.M], device=self.device)
                for n in range(self.N):
                    mu = self.mu.permute([1, 0])[n].expand(
                        ([self.M, self.M, self.Q]))
                    z = self.Xu[l].expand(([self.M, self.M, self.Q]))
                    z_t = z.permute([1, 0, 2])
                    alpha = self.w[l].expand(([self.M, self.M, self.Q]))
                    S = self.S.permute([1, 0])[n].expand(
                        ([self.M, self.M, self.Q]))
                    sigma = (self.sigma**2)[l].expand(([self.M, self.M]))
                    phi = self.phi[n, d, l].expand(([self.M, self.M]))

                    z_m = (z + z_t) / 2.0
                    dist_1 = (z - z_t)**2.0
                    dist_2 = (mu - z_m)**2.0
                    sub_denominator = 2.0 * alpha * S + 1

                    coe_1 = -0.25 * alpha
                    coe_2 = -alpha / sub_denominator

                    part_1 = torch.exp(
                        (dist_1 * coe_1 + dist_2 * coe_2).sum(dim=-1))
                    part_2 = torch.prod(1.0 / torch.sqrt(sub_denominator),
                                        dim=-1)
                    psi_mm += phi * sigma * part_1 * part_2
                psi_d.append(psi_mm)
            psi_d = torch.stack(psi_d, dim=0)
            psi.append((psi_d))
        psi = torch.stack(psi, dim=0)
        return psi

    def psi_2_loop(self):
        # self.phi = phi  # N * D * L
        # self.sigma = sigma**2  # L
        # self.w = w  # L *Q
        # self.mu = mu  # Q * N
        # self.S = S  # Q * N
        # self.Xu = Xu # L * M * Q
        psi = torch.zeros([self.L, self.D, self.M, self.M], device=self.device)
        for n in range(self.N):
            mu = self.mu.permute([1, 0])[n].expand(
                ([self.L, self.D, self.M, self.M, self.Q]))
            z = self.Xu.unsqueeze(1).unsqueeze(1).expand(
                ([self.L, self.D, self.M, self.M, self.Q]))
            z_t = z.permute([0, 1, 3, 2, 4])
            alpha = self.w.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(
                ([self.L, self.D, self.M, self.M, self.Q]))
            S = self.S.permute([1, 0])[n].expand(
                ([self.L, self.D, self.M, self.M, self.Q]))
            sigma = (self.sigma**2
                     ).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                         ([self.L, self.D, self.M, self.M]))
            phi = self.phi[n].permute([1,
                                       0]).unsqueeze(-1).unsqueeze(-1).expand(
                                           ([self.L, self.D, self.M, self.M]))

            z_m = (z + z_t) / 2.0
            dist_1 = (z - z_t)**2.0
            dist_2 = (mu - z_m)**2.0
            sub_denominator = 2.0 * alpha * S + 1

            coe_1 = -0.25 * alpha
            coe_2 = -alpha / sub_denominator

            part_1 = torch.exp((dist_1 * coe_1 + dist_2 * coe_2).sum(dim=-1))
            part_2 = torch.prod(1.0 / torch.sqrt(sub_denominator), dim=-1)
            psi += phi * sigma * part_1 * part_2

        return psi

    def psi_2_cpu(self):
        mu = self.mu.permute([1, 0]).unsqueeze(1).unsqueeze(1).expand(
            ([self.L, self.D, self.N, self.M, self.M, self.Q])).cpu()
        z = self.Xu.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(
            ([self.L, self.D, self.N, self.M, self.M, self.Q])).cpu()
        z_t = self.Xu.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(
            ([self.L, self.D, self.N, self.M, self.M,
              self.Q])).permute([0, 1, 2, 4, 3, 5]).cpu()
        alpha = self.w.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(
            1).expand(([self.L, self.D, self.N, self.M, self.M,
                        self.Q])).cpu()
        S = self.S.permute([1, 0]).unsqueeze(1).unsqueeze(1).expand(
            ([self.L, self.D, self.N, self.M, self.M, self.Q])).cpu()

        sigma = (self.sigma**2).unsqueeze(-1).unsqueeze(-1).unsqueeze(
            -1).unsqueeze(-1).expand(([self.L, self.D, self.N, self.M,
                                       self.M])).cpu()

        phi = self.phi.permute([2, 1, 0]).unsqueeze(-1).unsqueeze(-1).expand(
            ([self.L, self.D, self.N, self.M, self.M])).cpu()

        z_m = (z + z_t) / 2.0
        dist_1 = (z - z_t)**2.0
        dist_2 = (mu - z_m)**2.0
        sub_denominator = 2.0 * alpha * S + 1

        coe_1 = -0.25 * alpha
        coe_2 = -alpha / sub_denominator

        part_1 = torch.exp((dist_1 * coe_1 + dist_2 * coe_2).sum(dim=-1))
        part_2 = torch.prod(1.0 / torch.sqrt(sub_denominator), dim=-1)
        psi = (phi * sigma * part_1 * part_2).sum(dim=2)
        return psi.to(self.device)

    def psi_2f(self):
        # self.phi = phi  # N * D * L
        # self.sigma = sigma**2  # L
        # self.w = w  # L *Q
        # self.mu = mu  # Q * N
        # self.S = S  # Q * N * N
        # self.Xu = Xu # L * M * Q
        # self.Q = w.shape[1]
        # self.M = Xu.shape[1]

        mu = self.mu.permute([1, 0]).unsqueeze(1).unsqueeze(1).expand(
            ([self.D, self.N, self.M, self.M, self.Q])).cpu()

        # L
        z = self.Xu.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(
            ([self.L, self.D, self.N, self.M, self.M, self.Q])).cpu()
        # L
        z_t = self.Xu.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(
            ([self.L, self.D, self.N, self.M, self.M,
              self.Q])).permute([0, 1, 2, 4, 3, 5]).cpu()
        # L
        alpha = self.w.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(
            1).expand(([self.L, self.D, self.N, self.M, self.M,
                        self.Q])).cpu()

        S = torch.stack([torch.diag(self.S[i]) for i in range(self.Q)
                         ]).permute([1, 0]).unsqueeze(1).unsqueeze(1).expand(
                             ([self.D, self.N, self.M, self.M, self.Q])).cpu()

        # L
        sigma = (self.sigma**2).unsqueeze(-1).unsqueeze(-1).unsqueeze(
            -1).unsqueeze(-1).expand(([self.L, self.D, self.N, self.M,
                                       self.M])).cpu()

        # L
        phi = self.phi.permute([2, 1, 0]).unsqueeze(-1).unsqueeze(-1).expand(
            ([self.L, self.D, self.N, self.M, self.M])).cpu()

        psi = []
        for l in range(self.L):

            z_m = (z[l] + z_t[l]) / 2.0
            dist_1 = (z[l] - z_t[l])**2.0
            dist_2 = (mu - z_m)**2.0
            sub_denominator = 2.0 * alpha[l] * S + 1

            coe_1 = -0.25 * alpha[l]
            coe_2 = -alpha[l] / sub_denominator

            part_1 = torch.exp((dist_1 * coe_1 + dist_2 * coe_2).sum(dim=-1))
            part_2 = torch.prod(1.0 / torch.sqrt(sub_denominator), dim=-1)
            psi.append((phi[l] * sigma[l] * part_1 * part_2).sum(dim=1))
        psi = torch.stack(psi, dim=0)

        return psi.to(self.device)

        # print('psi.shape', psi.shape)
        # print(coe_2[0, 0, 0, 2, 2, 0])
        # print(alpha[0, 0, 0, 2, 2, 0])
        # print(z_t[0, 0, 0, 2, 2, 0])


class Psi_2D():
    def __init__(self, phi, sigma, w, mu, S, Xu, device):

        self.phi = phi  # N * D * L
        self.sigma = sigma.pow(2)  # L
        self.w = w  # L *Q
        self.mu = mu  # Q * N
        self.S = S  # Q * N
        self.N, self.D, self.L = self.phi.shape
        self.Xu = Xu  # L*M*Q
        self.Q = w.shape[1]
        self.M = Xu.shape[1]
        self.device = device

    def psi_1_nm(self, l=0, d=0):
        # phi   # N
        # sigma   # 1
        # w   # Q
        # mu  # Q * N
        # S   # Q * N
        # Xu  # M * Q

        mu = self.mu.permute([1, 0]).unsqueeze(1).expand(
            ([self.N, self.M, self.Q]))
        z = self.Xu[l].expand(([self.N, self.M, self.Q]))
        alpha = self.w[l].expand(([self.N, self.M, self.Q]))
        S = self.S.permute([1, 0]).unsqueeze(1).expand(
            ([self.N, self.M, self.Q]))
        sigma = self.sigma[l]
        phi = self.phi[:, d, l].unsqueeze(-1).expand(([self.N, self.M]))

        dist = (mu - z).pow(2)
        sub_denominator = alpha * S + 1.0
        coe_1 = -0.5 * alpha / sub_denominator
        part_1 = torch.exp((dist * coe_1).sum(dim=-1))
        part_2 = torch.prod(1.0 / torch.sqrt(sub_denominator), dim=-1)
        psi_nm = phi * sigma * part_1 * part_2

        return psi_nm

    def psi_2_mm(self, l=0, d=0):
        # self.phi = phi  # N * D * L
        # self.sigma = sigma.pow(2)  # L
        # self.w = w  # L *Q
        # self.mu = mu  # Q * N
        # self.S = S  # Q * N
        # self.Xu = Xu  # L*M*Q

        z = self.Xu[l].expand(([self.M, self.M, self.Q]))
        z_t = z.permute([1, 0, 2])
        alpha = self.w[l].expand(([self.M, self.M, self.Q]))
        sigma = self.sigma[l].pow(2)

        psi_mm = torch.zeros([self.M, self.M], device=self.device)

        for n in range(self.N):
            mu = self.mu[:, n].expand(([self.M, self.M, self.Q]))
            S = self.S[:, n].expand(([self.M, self.M, self.Q]))
            phi = self.phi[n, d, l]

            z_m = (z + z_t) / 2.0
            dist_1 = (z - z_t)**2.0
            dist_2 = (mu - z_m)**2.0
            sub_denominator = 2.0 * alpha * S + 1

            coe_1 = -0.25 * alpha
            coe_2 = -alpha / sub_denominator

            part_1 = torch.exp((dist_1 * coe_1 + dist_2 * coe_2).sum(dim=-1))
            part_2 = torch.prod(1.0 / torch.sqrt(sub_denominator), dim=-1)
            psi_mm += phi * sigma * part_1 * part_2

        return psi_mm


class Psi_2D_cpu():
    def __init__(self, phi, sigma, w, mu, S, Xu, device):

        self.phi = phi.cpu()  # N * D * L
        self.sigma = sigma.cpu().pow(2)  # L
        self.w = w.cpu()  # L *Q
        self.mu = mu.cpu()  # Q * N
        self.S = S  # Q * N
        self.N, self.D, self.L = self.phi.shape
        self.Xu = Xu.cpu()  # L*M*Q
        self.Q = w.shape[1]
        self.M = Xu.shape[1]
        self.device = device

    def psi_1_nm(self, l=0, d=0):
        # phi   # N
        # sigma   # 1
        # w   # Q
        # mu  # Q * N
        # S   # Q * N
        # Xu  # M * Q

        mu = self.mu.permute([1, 0]).unsqueeze(1).expand(
            ([self.N, self.M, self.Q])).cpu()
        z = self.Xu[l].expand(([self.N, self.M, self.Q])).cpu()
        alpha = self.w[l].expand(([self.N, self.M, self.Q])).cpu()
        S = self.S.permute([1, 0]).unsqueeze(1).expand(
            ([self.N, self.M, self.Q])).cpu()
        sigma = self.sigma[l].cpu()
        phi = self.phi[:, d, l].unsqueeze(-1).expand(([self.N, self.M])).cpu()

        dist = (mu - z)**2
        sub_denominator = alpha * S + 1.0
        coe_1 = -0.5 * alpha / sub_denominator
        part_1 = torch.exp((dist * coe_1).sum(dim=-1))
        part_2 = torch.prod(1.0 / torch.sqrt(sub_denominator), dim=-1)
        psi_nm = phi * sigma * part_1 * part_2

        return psi_nm.to(self.device)

    def psi_2_mm(self, l=0, d=0):
        # self.phi = phi  # N * D * L
        # self.sigma = sigma.pow(2)  # L
        # self.w = w  # L *Q
        # self.mu = mu  # Q * N
        # self.S = S  # Q * N
        # self.Xu = Xu  # L*M*Q
        # phi   # N
        # sigma   # 1
        # w   # Q
        # mu  # Q * N
        # S   # Q * N
        # Xu  # M * Q

        psi_mm = torch.zeros([self.M, self.M]).cpu()
        for n in range(self.N):
            mu = self.mu[:, n].expand(([self.M, self.M, self.Q])).cpu()
            z = self.Xu[l].expand(([self.M, self.M, self.Q])).cpu()
            z_t = z.permute([1, 0, 2]).cpu()
            alpha = self.w[l].expand(([self.M, self.M, self.Q])).cpu()
            S = self.S[:, n].expand(([self.M, self.M, self.Q])).cpu()
            sigma = self.sigma[l].cpu().pow(2)
            phi = self.phi[n, d, l].cpu()

            z_m = (z + z_t) / 2.0
            dist_1 = (z - z_t)**2.0
            dist_2 = (mu - z_m)**2.0
            sub_denominator = 2.0 * alpha * S + 1

            coe_1 = -0.25 * alpha
            coe_2 = -alpha / sub_denominator

            part_1 = torch.exp((dist_1 * coe_1 + dist_2 * coe_2).sum(dim=-1))
            part_2 = torch.prod(1.0 / torch.sqrt(sub_denominator), dim=-1)
            psi_mm += phi * sigma * part_1 * part_2

        return psi_mm.to(self.device)


def psi_1_nm(mu, Xu_l, w_l, S, phi_l_d, sigma_l):
    # phi   # N
    # sigma   # 1
    # w   # Q
    # mu  # Q * N
    # S   # Q * N
    # Xu  # M * Q
    n = mu.size(1)
    m = Xu_l.size(0)
    q = Xu_l.size(1)

    mu = mu.permute([1, 0]).unsqueeze(1).expand(([n, m, q]))
    z = Xu_l.expand(([n, m, q]))
    alpha = w_l.expand(([n, m, q]))
    S = S.permute([1, 0]).unsqueeze(1).expand(([n, m, q]))
    sigma = sigma_l
    phi = phi_l_d.unsqueeze(-1).expand(([n, m]))

    dist = (mu - z).pow(2)
    sub_denominator = alpha * S + 1.0
    coe_1 = -0.5 * alpha / sub_denominator
    part_1 = torch.exp((dist * coe_1).sum(dim=-1))
    part_2 = torch.prod(1.0 / torch.sqrt(sub_denominator), dim=-1)
    psi_nm = phi * sigma * part_1 * part_2

    return psi_nm


def psi_2_mm(mu_n, Xu_l, w_l, S_n, phi_l_d, sigma_l):
    # self.phi = phi  # N * D * L
    # self.sigma = sigma.pow(2)  # L
    # self.w = w  # L *Q
    # self.mu = mu  # Q * N
    # self.S = S  # Q * N
    # self.Xu = Xu  # L*M*Q

    NN = mu_n.size(1)
    m = Xu_l.size(0)
    q = Xu_l.size(1)

    z = Xu_l.unsqueeze(1)
    z_t = Xu_l.unsqueeze(0)
    alpha = w_l.unsqueeze(0).unsqueeze(0)
    sigma = sigma_l.pow(2)

    psi_mm = torch.zeros([m, m], device=mu_n.device)

    for n in range(NN):
        mu = mu_n[:, n].unsqueeze(0).unsqueeze(0)
        S = S_n[:, n].unsqueeze(0).unsqueeze(0)
        phi = phi_l_d[n]

        z_m = (z + z_t) / 2.0
        dist_1 = (z - z_t).pow(2)
        dist_2 = (mu - z_m).pow(2)
        sub_denominator = 2.0 * alpha * S + 1

        coe_1 = -0.25 * alpha
        coe_2 = -alpha / sub_denominator

        part_1 = torch.exp((dist_1 * coe_1 + dist_2 * coe_2).sum(dim=-1))
        part_2 = torch.prod(1.0 / torch.sqrt(sub_denominator), dim=-1)
        psi_mm += phi * sigma * part_1 * part_2

    return psi_mm


def psi_2_mm_cpu(mu_n, Xu_l, w_l, S_n, phi_l_d, sigma_l):
    # self.phi = phi  # N * D * L
    # self.sigma = sigma.pow(2)  # L
    # self.w = w  # L *Q
    # self.mu = mu  # Q * N
    # self.S = S  # Q * N
    # self.Xu = Xu  # L*M*Q

    NN = mu_n.size(1)
    m = Xu_l.size(0)
    q = Xu_l.size(1)

    z = Xu_l.expand(([m, m, q])).cpu()
    z_t = z.permute([1, 0, 2]).cpu()
    alpha = w_l.expand(([m, m, q])).cpu()
    sigma = sigma_l.pow(2).cpu()

    psi_mm = torch.zeros([m, m]).cpu()

    for n in range(NN):
        mu = mu_n[:, n].expand(([m, m, q])).cpu()
        S = S_n[:, n].expand(([m, m, q])).cpu()
        phi = phi_l_d[n].cpu()

        z_m = (z + z_t) / 2.0
        dist_1 = (z - z_t).pow(2)
        dist_2 = (mu - z_m).pow(2)
        sub_denominator = 2.0 * alpha * S + 1

        coe_1 = -0.25 * alpha
        coe_2 = -alpha / sub_denominator

        part_1 = torch.exp((dist_1 * coe_1 + dist_2 * coe_2).sum(dim=-1))
        part_2 = torch.prod(1.0 / torch.sqrt(sub_denominator), dim=-1)
        psi_mm += phi * sigma * part_1 * part_2

    return psi_mm.to(mu_n.device)


# -------------------new--------------------------
def psi1_nm(mu, Xu_l, w_l, S, phi_l_d, sigma_l):
    # phi   # N
    # sigma   # 1
    # w   # Q
    # mu  # Q * N
    # S   # Q * N
    # Xu  # M * Q
    # n = mu.size(1)
    # m = Xu_l.size(0)
    # q = Xu_l.size(1)

    mu = mu.t().unsqueeze(1)  # n * 1 * q
    z = Xu_l.unsqueeze(0)  # 1 * m * q
    alpha = w_l.unsqueeze(0).unsqueeze(0)  # 1 * 1 * q
    S = S.t().unsqueeze(1)  # n * 1 * q
    sigma = sigma_l
    phi = phi_l_d.unsqueeze(-1)  # n *  1

    dist = (mu - z).pow(2)
    sub_denominator = alpha * S + 1.0
    coe_1 = -0.5 * alpha / sub_denominator
    part_1 = torch.exp((dist * coe_1).sum(dim=-1))
    part_2 = torch.prod(1.0 / torch.sqrt(sub_denominator), dim=-1)
    psi_nm = phi * sigma * part_1 * part_2

    return psi_nm


def psi2_mm(mu_n, Xu_l, w_l, S_n, phi_l_d, sigma_l):
    # self.phi = phi  # N * D * L
    # self.sigma = sigma.pow(2)  # L
    # self.w = w  # L *Q
    # self.mu = mu  # Q * N
    # self.S = S  # Q * N
    # self.Xu = Xu  # L*M*Q

    # NN = mu_n.size(1)
    # m = Xu_l.size(0)
    # q = Xu_l.size(1)

    z = Xu_l.unsqueeze(1).unsqueeze(0)  # 1 * m * 1 * q
    z_t = Xu_l.unsqueeze(0).unsqueeze(0)  # 1 * 1 * m * q
    alpha = w_l.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # 1 * 1 * 1 * q
    mu = mu_n.t().unsqueeze(1).unsqueeze(1)  # n * 1 * 1 * q
    S = S_n.t().unsqueeze(1).unsqueeze(1)  # n * 1 * 1 * q
    sigma = sigma_l.pow(2)
    phi = phi_l_d.unsqueeze(-1).unsqueeze(-1)  # n * 1 * 1

    # psi_mm = torch.zeros([m, m], device=mu_n.device)

    # for n in range(NN):

    z_m = (z + z_t) / 2.0
    dist_1 = (z - z_t).pow(2)
    dist_2 = (mu - z_m).pow(2)
    sub_denominator = 2.0 * alpha * S + 1

    coe_1 = -0.25 * alpha
    coe_2 = -alpha / sub_denominator

    part_1 = torch.exp((dist_1 * coe_1 + dist_2 * coe_2).sum(dim=-1))
    part_2 = torch.prod(1.0 / torch.sqrt(sub_denominator), dim=-1)
    psi_mm = (phi * sigma * part_1 * part_2).sum(dim=0)

    return psi_mm


if __name__ == '__main__':

    from time import *
    from scipy.io import loadmat
    device = torch.device('cuda')

    phi_kern = loadmat("./../datasets/tt_psi.mat")
    mu_S = loadmat("./../datasets/tt_mu_S.mat")
    Xu = loadmat("./../datasets/tt_toy1_Xu.mat")
    phi = [torch.from_numpy(phi_kern['phidd'][0, i]) for i in range(3)]
    phi = torch.stack(phi, dim=0).permute([1, 2, 0]).to(device)
    sigma = torch.ones((3)).to(device)
    w = [
        torch.from_numpy(phi_kern['mytestkern'][0, i][0, 0][3])
        for i in range(3)
    ]
    w = torch.stack(w, dim=0).squeeze().to(device)
    mu = torch.from_numpy(mu_S['mu']).t().to(device)
    S = torch.from_numpy(mu_S['S']).t().to(device)
    # S = torch.stack([torch.diag(S[i]) for i in range(8)], dim=0).to(device)
    Xu = [torch.from_numpy(Xu['tt_Xu'][0, i]) for i in range(3)]
    Xu = torch.stack(Xu, dim=0).to(device)

    # mu = self.mu.permute([1, 0]).expand([self.L, self.D, self.N, self.M, self.Q])
    # mu = mu.permute([1, 0]).expand([3, 30, 50, 100, 8])
    # mu = mu.expand([3, 30, 50, 100, 8])
    # print(mu.shape)
    # print(mu)

    # print(sigma.shape)

    # self.M.to(self.device)
    psi = Psi(phi=phi, sigma=sigma, w=w, mu=mu, S=S, Xu=Xu, device=device)
    psi_2d = Psi_2D(phi=phi,
                    sigma=sigma,
                    w=w,
                    mu=mu,
                    S=S,
                    Xu=Xu,
                    device=device)
    # psi_0 = psi.psi_0().to(device)
    # psi_1 = psi.psi_1().to(device)
    # psi_2 = psi.psi_2().to(device)

    # bt = time()
    # psi_2 = psi.psi_2()
    # et = time()
    # print('time of no loop is : ')
    # print(et - bt)
    # bt = time()
    # psi_2f = psi.psi_2f()
    # et = time()
    # print('time of loop is : ')
    # print(et - bt)
    # # print(psi_0.shape)
    # # print(psi_1.shape)
    # # print(psi_2.shape)
    # bt = time()
    # p2 = psi.psi_2()
    # et = time()
    # print('time of p2 is : ')
    # print(et - bt)

    bt = time()
    p2_loop = psi.psi_2_loop()
    et = time()
    print('time of p2_loop is : ')
    print(et - bt)

    bt = time()
    p22 = psi.psi_22()
    et = time()
    print('time of p22 is : ')
    print(et - bt)

    bt = time()
    p1 = psi.psi_1()
    et = time()
    print('time of p1 is : ')
    print(et - bt)

    bt = time()
    p11 = psi.psi_11()
    et = time()
    print('time of p11 is : ')
    print(et - bt)
    print(p2_loop[0, 0, 10, 10])
    print(p22[0, 0, 10, 10])
    print(p2_loop[0, 0, 7, 7])
    print(p22[0, 0, 7, 7])
    print(p1[1, 2, 6, 15])
    print(p11[1, 2, 6, 15])
    print('******************psi_1*************************')
    print(psi1_nm(mu, Xu[1], w[1], S, phi[:, 2, 1], sigma[1])[6, 15])
    print(psi_2d.psi_1_nm(1, 2)[6, 15])
    print('******************psi_2*************************')
    print(psi2_mm(mu, Xu[0], w[0], S, phi[:, 0, 0], sigma[0])[10, 10])
    print(psi_2d.psi_2_mm()[10, 10])
    print(psi2_mm(mu, Xu[0], w[0], S, phi[:, 0, 0], sigma[0])[7, 7])
    print(psi_2d.psi_2_mm()[7, 7])
    print('psi_2.shape', psi_2d.psi_2_mm().shape)

    # pp = psi_2d.psi_2_mm()
    pp = psi_2_mm(mu, Xu[0], w[0], S, phi[:, 0, 0], sigma[0])
    flag = True
    for i in range(pp.shape[0]):
        for j in range(pp.shape[1]):
            if pp[i, j] != pp.t()[i, j]:
                flag = False
    print('is Ture ?', flag)
