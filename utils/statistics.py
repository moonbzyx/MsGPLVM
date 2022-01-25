import torch
# import numpy as np
from scipy.io import loadmat
# import matplotlib.pyplot as plt


class Psi():
    def __init__(self, phi, sigma, w, mu, S, Xu, device):

        self.phi = phi  # N * D * L
        self.sigma = sigma**2  # L
        self.w = w  # L *Q
        self.mu = mu  # Q * N
        self.S = S  # Q * N * N
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
        S = torch.stack([torch.diag(self.S[i]) for i in range(self.Q)
                         ]).permute([1, 0]).unsqueeze(1).expand(
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
        S = torch.stack([torch.diag(self.S[i]) for i in range(self.Q)
                         ]).permute([1, 0]).unsqueeze(1).unsqueeze(1).expand(
                             ([self.L, self.D, self.N, self.M, self.M,
                               self.Q]))
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

    def psi_2_loop(self):
        # self.phi = phi  # N * D * L
        # self.sigma = sigma**2  # L
        # self.w = w  # L *Q
        # self.mu = mu  # Q * N
        # self.S = S  # Q * N * N
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
            S = torch.stack([torch.diag(self.S[i]) for i in range(self.Q)
                             ]).permute([1, 0])[n].expand(
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
        S = torch.stack([torch.diag(self.S[i]) for i in range(self.Q)
                         ]).permute([1, 0]).unsqueeze(1).unsqueeze(1).expand(
                             ([self.L, self.D, self.N, self.M, self.M,
                               self.Q])).cpu()

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


if __name__ == '__main__':

    from time import *
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
    S = torch.from_numpy(mu_S['S']).t()
    S = torch.stack([torch.diag(S[i]) for i in range(8)], dim=0).to(device)
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
    # # print(psi_2[0, 0, 10, 10])
    # # print(psi_2[0, 0, 7, 7])
    # print(psi_2f[0, 0].shape)
    bt = time()
    p = psi.psi_2_loop()
    et = time()
    print('time of loop is : ')
    print(et - bt)

    bt = time()
    p1 = psi.psi_1()
    et = time()
    print('time of no loop p1 is : ')
    print(et - bt)
