import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from scipy import linalg


def doa_root_music(covmat, nsig, spacing=0.5):
    """
    Estimate arrival directions of signals using root-MUSIC for a uniform
    linear array (ULA)

    :param numpy.2darray covmat:
        Sensor covariance matrix, specified as a complex-valued, positive-
        definite M-by-M matrix. The quantity M is the number of elements
        in the ULA array
    :param int nsig:
        Number of arriving signals, specified as a positive integer. The
        number of signals must be smaller than the number of elements in
        the ULA array
    :param float spacing:
        Distance (wavelength) between array elements. ``default 0.5``

    :return: doa angles in degrees
    :rtype: list
    """

    n_covmat = np.shape(covmat)[0]

    _, eig_vects = linalg.eigh(covmat)
    noise_subspace = eig_vects[:, :-nsig]

    # Compute the coefficients for the polynomial.
    noise_mat = noise_subspace @ noise_subspace.T.conj()
    coeff = np.zeros((n_covmat - 1,), dtype=np.complex_)
    for i in range(1, n_covmat):
        coeff[i - 1] = np.trace(noise_mat, i)
    coeff = np.hstack((coeff[::-1], np.trace(noise_mat), coeff.conj()))

    roots = np.roots(coeff)

    # Find k points inside the unit circle that are also closest to the unit
    # circle.
    mask = np.abs(roots) <= 1
    # On the unit circle. Need to find the closest point and remove it.
    for _, i in enumerate(np.where(np.abs(roots) == 1)[0]):
        mask_idx = np.argsort(np.abs(roots - roots[i]))[1]
        mask[mask_idx] = False

    roots = roots[mask]
    sorted_indices = np.argsort(1.0 - np.abs(roots))
    sin_vals = np.angle(roots[sorted_indices[:nsig]]) / (2 * np.pi * spacing)

    return np.degrees(np.arcsin(sin_vals))


class DOARootMusic(nn.Module):
    def __init__(self, nsig, spacing=0.5):
        super(DOARootMusic, self).__init__()
        self.nsig = nsig
        self.spacing = spacing

    def forward(self, covmat):
        n_covmat = covmat.shape[0]

        _, eig_vects = torch.linalg.eigh(covmat)
        noise_subspace = eig_vects[:, : -self.nsig]

        # Compute the coefficients for the polynomial.
        noise_mat = torch.matmul(noise_subspace, noise_subspace.T.conj())
        coeff = torch.zeros((n_covmat - 1,), dtype=torch.complex128)
        for i in range(1, n_covmat):
            coeff[i - 1] = torch.trace(noise_mat[i:])
        coeff = torch.cat(
            (torch.flip(coeff, [0]), torch.tensor([torch.trace(noise_mat)]), coeff.conj())
        )

        roots = torch.roots(coeff)

        # Find k points inside the unit circle that are also closest to the unit
        # circle.
        mask = torch.abs(roots) <= 1
        # On the unit circle. Need to find the closest point and remove it.
        for _, i in enumerate(torch.where(torch.abs(roots) == 1)[0]):
            mask_idx = torch.argsort(torch.abs(roots - roots[i]))[1]
            mask[mask_idx] = False

        roots = roots[mask]
        sorted_indices = torch.argsort(1.0 - torch.abs(roots))
        sin_vals = torch.angle(roots[sorted_indices[: self.nsig]]) / (2 * torch.pi * self.spacing)

        return torch.degrees(torch.asin(sin_vals))
