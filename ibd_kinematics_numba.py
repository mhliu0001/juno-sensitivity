import numpy as np
import scipy.constants
import scipy.integrate
import numba
from numba import njit


# ------------------------------
# Numba-accelerated kernels
# ------------------------------


@njit(cache=True)
def _calc_ABC_kernel(t, f1, f2, g1, g2, me2, M2, Delta2, M, Delta):
    """
    Numba kernel for calc_ABC.
    t, f1, f2, g1, g2 can be scalars or arrays (NumPy supported by Numba).
    """
    # Term 16A:
    group1_inner = (
        4 * f1**2 * (4 * M2 + t + me2)
        + 4 * g1**2 * (-4 * M2 + t + me2)
        + f2**2 * (t**2 / M2 + 4 * t + 4 * me2)
        + (4 * me2 * t * g2**2) / M2
        + 8 * f1 * f2 * (2 * t + me2)
        + 16 * me2 * g1 * g2
    )
    term1 = (t - me2) * group1_inner

    group2_inner = (
        (4 * f1**2 + t * f2**2 / M2) * (4 * M2 + t - me2)
        + 4 * g1**2 * (4 * M2 - t + me2)
        + (4 * me2 * g2**2 * (t - me2)) / M2
        + 8 * f1 * f2 * (2 * t - me2)
        + 16 * me2 * g1 * g2
    )
    term2 = -Delta2 * group2_inner

    term3 = -32 * me2 * M * Delta * g1 * (f1 + f2)

    A = (term1 + term2 + term3) / 16.0

    # Term 16B:
    termB_num = (
        16 * t * g1 * (f1 + f2)
        + (4 * me2 * Delta * (f2**2 + f1 * f2 + 2 * g1 * g2)) / M
    )
    B = termB_num / 16.0

    # Term 16C:
    termC_num = 4 * (f1**2 + g1**2) - (t * f2**2) / M2
    C = termC_num / 16.0

    return A, B, C


@njit(cache=True)
def _get_kinematic_bounds_kernel(E_nu, m_p, m_n, m_e, delta):
    """
    Numba kernel for get_kinematic_bounds.
    E_nu must be an array here (wrapper will ensure that).
    """
    E_e_min = np.zeros_like(E_nu)
    E_e_max = np.zeros_like(E_nu)

    E_nu_threshold = ((m_n + m_e) ** 2 - m_p**2) / (2 * m_p)
    below_threshold = E_nu < E_nu_threshold

    E_nu_valid = E_nu[~below_threshold]
    if E_nu_valid.size == 0:
        return E_e_min, E_e_max

    s = m_p**2 + 2 * m_p * E_nu_valid
    sqrt_s = np.sqrt(s)
    E_nu_cm = (s - m_p**2) / (2 * sqrt_s)
    E_e_cm = (s - m_n**2 + m_e**2) / (2 * sqrt_s)
    p_e_cm = np.sqrt(E_e_cm**2 - m_e**2)

    E_e_min[~below_threshold] = E_nu_valid - delta - E_nu_cm * (E_e_cm + p_e_cm) / m_p
    E_e_max[~below_threshold] = E_nu_valid - delta - E_nu_cm * (E_e_cm - p_e_cm) / m_p

    return E_e_min, E_e_max


@njit(cache=True)
def _differential_cross_section_kernel(
    E_nu,
    E_e,
    m_p,
    m_n,
    m_e,
    m_pi,
    M,
    M_V2,
    M_A2,
    G_F,
    cos_theta_c,
    alpha,
    hbar_c,
    g_1_0,
    xi,
    delta,
):
    """
    Numba kernel for the core of differential_cross_section.

    E_nu, E_e: arrays (broadcasted in the wrapper).
    Returns an array of the same shape as E_e.
    """
    # Kinematic bounds (vectorized)
    E_e_min, E_e_max = _get_kinematic_bounds_kernel(E_nu, m_p, m_n, m_e, delta)

    dsigma_dE_corrected = np.zeros_like(E_e)

    threshold = ((m_n + m_e) ** 2 - m_p**2) / (2 * m_p)
    kinematic_mask = (E_e >= E_e_min) & (E_e <= E_e_max) & (E_nu >= threshold)

    E_nu_valid = E_nu[kinematic_mask]
    E_e_valid = E_e[kinematic_mask]
    if E_nu_valid.size == 0:
        return dsigma_dE_corrected

    # Mandelstam variables
    s = m_p**2 + 2 * m_p * E_nu_valid
    s_minus_u = 2 * m_p * (E_nu_valid + E_e_valid) - m_e**2
    t = m_n**2 - m_p**2 - 2 * m_p * (E_nu_valid - E_e_valid)

    # Form factors
    t_div_MV2 = t / M_V2
    t_div_MA2 = t / M_A2
    DV = 1.0 / (1 - t_div_MV2) ** 2
    DA = 1.0 / (1 - t_div_MA2) ** 2

    tau = t / (4 * M**2)
    f1 = DV * (1 - (1 + xi) * tau) / (1 - tau)
    f2 = DV * xi / (1 - tau)
    g1 = g_1_0 * DA
    g2 = 2 * M**2 * g1 / (m_pi**2 - t)

    me2 = m_e**2
    M2 = M**2
    Delta = m_n - m_p
    Delta2 = Delta**2

    # A, B, C
    A, B, C = _calc_ABC_kernel(t, f1, f2, g1, g2, me2, M2, Delta2, M, Delta)
    M2_elem = A - s_minus_u * B + s_minus_u**2 * C

    # Differential cross section (Eq. 5 + Eq. 11 factor)
    dsigma_dt = (G_F**2 * cos_theta_c**2) / (2 * np.pi * (s - m_p**2) ** 2) * M2_elem
    dsigma_dE = 2 * m_p * dsigma_dt

    # Radiative correction
    rad = 1.0 + (alpha / np.pi) * (
        6.0 + 1.5 * np.log(m_p / (2 * E_e_valid)) + 1.2 * (m_e / E_e_valid) ** 1.5
    )

    # To cm^2
    dsigma_dE_corrected[kinematic_mask] = dsigma_dE * rad * (hbar_c**2)

    return dsigma_dE_corrected


# ------------------------------
# Public class
# ------------------------------


class StrumiaVissani:
    def __init__(self):
        # --- Constants (MeV) ---
        self.G_F = 1.16637e-11  # MeV^-2
        self.cos_theta_c = 0.9746
        self.m_p = 938.272
        self.m_n = 939.565
        self.m_e = 0.510999
        self.m_pi = 139.570
        self.alpha = 1.0 / 137.036
        self.M = (self.m_p + self.m_n) / 2.0
        self.Delta = self.m_n - self.m_p
        self.delta = (self.m_n**2 - self.m_p**2 - self.m_e**2) / (2 * self.m_p)
        self.hbar_c = (
            scipy.constants.hbar * scipy.constants.c / scipy.constants.e / 1e4
        )  # MeV*cm

        # Precomputed squares for reuse
        self.me2 = self.m_e**2
        self.M2 = self.M**2
        self.Delta2 = self.Delta**2

        # Form Factors
        self.M_V2 = 0.71 * 1e6  # MeV^2
        self.M_A2 = 1.0 * 1e6  # MeV^2
        self.g_1_0 = -1.27  # g_1 at t=0
        self.mu_p = 2.79284734463  # Proton magnetic moment, CODATA 2022
        self.mu_n = -1.91304276  # Neutron magnetic moment, CODATA 2022
        self.xi = self.mu_p - self.mu_n - 1.0  # Anomalous magnetic moment difference

    # --------------------------
    # Kinematic bounds (wrapper)
    # --------------------------
    def get_kinematic_bounds(self, E_nu):
        """
        Calculates the kinematic bounds for the outgoing positron energy E_e
        given the incoming neutrino energy E_nu.
        """
        E_nu = np.atleast_1d(E_nu)
        return _get_kinematic_bounds_kernel(
            E_nu, self.m_p, self.m_n, self.m_e, self.delta
        )

    # --------------------------
    # A, B, C coefficients
    # --------------------------
    def calc_ABC(self, t, f1, f2, g1, g2):
        """
        Calculates A, B, C coefficients using Eq. 6 from Strumia-Vissani.
        Wrapper around the Numba kernel.
        """
        return _calc_ABC_kernel(
            t, f1, f2, g1, g2, self.me2, self.M2, self.Delta2, self.M, self.Delta
        )

    # --------------------------
    # Differential cross section
    # --------------------------
    def differential_cross_section(self, E_nu, E_e):
        """
        Calculates the differential cross section dsigma/dE_e for IBD using the
        exact Strumia-Vissani formula (Eq. 11).
        """
        E_nu = np.atleast_1d(E_nu)
        E_e = np.atleast_1d(E_e)
        E_nu, E_e = np.broadcast_arrays(E_nu, E_e)

        return _differential_cross_section_kernel(
            E_nu,
            E_e,
            self.m_p,
            self.m_n,
            self.m_e,
            self.m_pi,
            self.M,
            self.M_V2,
            self.M_A2,
            self.G_F,
            self.cos_theta_c,
            self.alpha,
            self.hbar_c,
            self.g_1_0,
            self.xi,
            self.delta,
        )

    # --------------------------
    # E_dep PDF
    # --------------------------
    def get_E_dep_pdf(self, E_nu, E_dep):
        """
        Calculates the probability density function (PDF) of deposited energy
        E_dep for a given incoming neutrino energy E_nu.
        """
        E_nu = np.atleast_1d(E_nu)
        E_dep = np.atleast_1d(E_dep)

        # E_dep = E_e + m_e
        sigma = self.differential_cross_section(E_nu, E_dep - self.m_e)

        # Normalize by the total cross section
        sigma_total = self.get_total_cross_section(E_nu)

        # NOTE: This preserves your original logic, including the potential
        # array-scalar comparison. If you only ever pass scalar E_nu, this
        # behaves as before.
        if sigma_total == 0:
            pdf = np.zeros_like(E_dep)
        else:
            pdf = sigma / sigma_total
        return pdf

    # --------------------------
    # Total cross section
    # --------------------------
    def get_total_cross_section(self, E_nu):
        """
        Calculates the total cross section for IBD by integrating the
        differential cross section over allowed positron energies.
        """
        E_nu = np.atleast_1d(E_nu)
        E_min, E_max = self.get_kinematic_bounds(E_nu)

        total_sigma = np.zeros_like(E_nu)
        for i, E_nu_val in enumerate(E_nu):
            E_min_val, E_max_val = E_min[i], E_max[i]
            if E_max_val <= E_min_val:
                total_sigma[i] = 0.0
                continue

            # quad will call the Python wrapper which uses the Numba kernel inside
            total_sigma[i], abs_error = scipy.integrate.quad(
                lambda E_e: float(self.differential_cross_section(E_nu_val, E_e)),
                E_min_val,
                E_max_val,
            )

            if total_sigma[i] > 0 and abs_error / total_sigma[i] > 1e-5:
                print(
                    f"Warning: High relative error in integration: "
                    f"{abs_error / total_sigma[i]:.2e}"
                )
        return total_sigma
