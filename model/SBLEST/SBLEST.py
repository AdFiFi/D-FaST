import torch
import torch.nn as nn
import numpy as np
from ..base import BaseConfig, ModelOutputs


class SBLESTConfig(BaseConfig):
    def __init__(self,
                 node_size,
                 node_feature_size,
                 time_series_size,
                 num_classes,
                 num_kernels=25):
        super(SBLESTConfig, self).__init__(node_size=node_size,
                                           node_feature_size=node_feature_size,
                                           time_series_size=time_series_size,
                                           num_classes=num_classes)
        self.tau = 1
        self.K = 5


class EnhancedCovariance(nn.Module):
    def __init__(self, config: SBLESTConfig):
        super().__init__()
        self.K = config.K
        self.tau = config.tau

    def forward(self, X, Wh=None):
        # Initialization, [KC, KC]: dimension of augmented covariance matrix
        X_order_k = None
        C, T, M = X.shape
        Cov = []
        Sig_Cov = torch.zeros(self.K * C, self.K * C, device=X.device)

        for m in range(M):
            X_m = X[:, :, m]
            X_m_hat = torch.DoubleTensor().to(X.device)

            # Generate augmented EEG data
            for k in range(self.K):
                n_delay = k * self.tau
                if n_delay == 0:
                    X_order_k = X_m.clone()
                else:
                    X_order_k[:, 0:n_delay] = 0
                    X_order_k[:, n_delay:T] = X_m[:, 0:T - n_delay].clone()
                X_m_hat = torch.cat((X_m_hat, X_order_k), 0)

            # Compute covariance matrices
            R_m = torch.mm(X_m_hat, X_m_hat.T)

            # Trace normalization
            R_m = R_m / R_m.trace()
            Cov.append(R_m)

            Sig_Cov = Sig_Cov + R_m

        # Compute Whitening matrix (Rp).
        if Wh is None:
            Wh = Sig_Cov / M
        # else:
        #     Wh = Wh + Sig_Cov / M * 0.0001


        # Whitening, logarithm transform, and Vectorization
        Cov_whiten = torch.zeros(M, self.K * C, self.K * C, dtype=torch.float64, device=X.device)
        R_train = torch.zeros(M, self.K * C * self.K * C, dtype=torch.float64, device=X.device)

        for m in range(M):
            # progress_bar(m, M)

            # whitening
            Wh_inverse = self.matrix_operations(Wh)  # Rp^(-1/2)
            temp_cov = Wh_inverse @ Cov[m] @ Wh_inverse
            Cov_whiten[m, :, :] = (temp_cov + temp_cov.T) / 2
            R_m = self.logm(Cov_whiten[m, :, :])
            R_m = R_m.reshape(R_m.numel())  # column-wise vectorization
            R_train[m, :] = R_m

        return R_train, Wh

    @staticmethod
    def matrix_operations(A):
        """Calculate the -1/2 power of matrix A"""

        V, Q = torch.linalg.eig(A)
        V_inverse = torch.diag(V ** (-0.5))
        A_inverse = torch.mm(torch.mm(Q, V_inverse), torch.linalg.inv(Q))

        return A_inverse.double()

    @staticmethod
    def logm(A):
        """Calculate the matrix logarithm of matrix A"""

        V, Q = torch.linalg.eig(A)  # V为特征值,Q为特征向量
        V_log = torch.diag(torch.log(V))
        A_logm = torch.mm(torch.mm(Q, V_log), torch.linalg.inv(Q))

        return A_logm.double()


class SBLEST(nn.Module):
    """
    A Reproduction of SBLEST:
    Mostly based on https://github.com/EEGdecoding/Code-SBLEST.

    Sparse Bayesian Learning for End-to-End EEG Decoding

    W. Wang, F. Qi, D. Wipf, C. Cai, T. Yu, Y. Li, et al.

    IEEE Transactions on Pattern Analysis and Machine Intelligence 2023
    """
    def __init__(self, config: SBLESTConfig):
        super(SBLEST, self).__init__()
        self.config = config
        self.enhance_conv = EnhancedCovariance(config)
        self.Wh = None
        self.W = None

    def forward(self, time_series, labels):

        # Compute enhanced covariance matrices and whitening matrix
        R_train, self.Wh = self.enhance_conv(time_series, self.Wh)
        # print('\n')

        # Check properties of R
        M, D_R = R_train.shape  # M: number of samples; D_R: dimension of vec(R_m)
        KC = round(np.sqrt(D_R))
        Loss_old = 1e12
        threshold = 0.05
        r2_list = []

        assert D_R == KC ** 2, "ERROR: Columns of A do not align with square matrix"

        # Check if R is symmetric
        for j in range(M):
            row_cov = torch.reshape(R_train[j, :], (KC, KC))
            row_cov = (row_cov + row_cov.T) / 2
            assert torch.norm(row_cov - row_cov.T) < 1e-4, "ERROR: Measurement row does not form symmetric matrix"

        # Initializations
        # estimated low-rank matrix W initialized to be Zeros
        U = torch.zeros(KC, KC, dtype=torch.float64, device=time_series.device)
        # covariance matrix of Gaussian prior distribution is initialized to be unit diagonal matrix
        Psi = torch.eye(KC, dtype=torch.float64, device=time_series.device)
        lambda_noise = 1.  # variance of the additive noise set to 1

        # Optimization loop
        for i in range(1000 + 1):

            # update B,Sigma_y,u
            RPR = torch.zeros(M, M, dtype=torch.float64, device=time_series.device)
            B = torch.zeros(KC ** 2, M, dtype=torch.float64, device=time_series.device)
            for j in range(KC):
                start = j * KC
                stop = start + KC
                Temp = torch.mm(Psi, R_train[:, start:stop].T)
                B[start:stop, :] = Temp
                RPR = RPR + torch.mm(R_train[:, start:stop], Temp)
            Sigma_y = RPR + lambda_noise * torch.eye(M, dtype=torch.float64, device=time_series.device)
            uc = torch.mm(torch.mm(B, torch.inverse(Sigma_y)), labels)  # maximum a posterior estimation of uc
            Uc = torch.reshape(uc, (KC, KC))
            U = (Uc + Uc.T) / 2
            u = U.T.flatten()  # vec operation (Torch)

            # Update Phi (dual variable of Psi)
            Phi = []
            SR = torch.mm(torch.inverse(Sigma_y), R_train)
            for j in range(KC):
                start = j * KC
                stop = start + KC
                Phi_temp = Psi - Psi @ R_train[:, start:stop].T @ SR[:, start:stop] @ Psi
                Phi.append(Phi_temp)

            # Update Psi
            PHI = 0
            UU = 0
            for j in range(KC):
                PHI = PHI + Phi[j]
                UU = UU + U[:, j].reshape(-1, 1) @ U[:, j].reshape(-1, 1).T
            # UU = U @ U.T
            Psi = ((UU + UU.T) / 2 + (PHI + PHI.T) / 2) / KC  # make sure Psi is symmetric

            # Update theta (dual variable of lambda)
            theta = 0
            for j in range(KC):
                start = j * KC
                stop = start + KC
                theta = theta + (Phi[j] @ R_train[:, start:stop].T @ R_train[:, start:stop]).trace()

            # Update lambda
            lambda_noise = ((torch.norm(labels - (R_train @ u).reshape(-1, 1), p=2) ** 2).sum() + theta) / M

            # Convergence check
            Loss = labels.T @ torch.inverse(Sigma_y) @ labels + torch.log(torch.det(Sigma_y))
            delta_loss = abs(Loss_old - Loss.cpu().numpy()) / abs(Loss_old)
            if delta_loss < 2e-4:
                print('EXIT: Change in loss below threshold')
                break
            Loss_old = Loss.cpu().numpy()
            if i % 100 == 99:
                print('Iterations: ', str(i + 1), '  lambda: ', str(lambda_noise.cpu().numpy()), '  Loss: ',
                      float(Loss.cpu().numpy()), '  Delta_Loss: ', float(delta_loss))

        # Eigen-decomposition of W
        self.W = U
        D, V_all = torch.linalg.eig(self.W)
        D, V_all = D.double().cpu().numpy(), V_all.double().cpu().numpy()
        idx = D.argsort()
        D = D[idx]
        V_all = V_all[:, idx]  # each column of V represents a spatio-temporal filter
        alpha_all = D

        # Determine spatio-temporal filters V and classifier weights alpha
        d = np.abs(alpha_all)
        d_max = np.max(d)
        w_norm = d / d_max  # normalize eigenvalues of W by the maximum eigenvalue
        index = np.where(w_norm > threshold)[
            0]  # indices of selected V according to a pre-defined threshold,.e.g., 0.05
        V = V_all[:, index]
        alpha = alpha_all[index]

        return self.W, alpha, V, self.Wh
