import numpy as np
import scipy
from tqdm import tqdm


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
        self.delta = (self.m_n**2 - self.m_p**2 - self.m_e**2) / 2 / self.m_p
        self.hbar_c = (
            scipy.constants.hbar * scipy.constants.c / scipy.constants.e / 1e4
        )  # MeV*cm

        # Form Factors
        self.M_V2 = 0.71 * 1e6  # MeV^2
        self.M_A2 = 1.0 * 1e6  # MeV^2
        self.g_1_0 = -1.27  # g_1 at t=0
        self.mu_p = 2.79284734463  # Proton magnetic moment, CODATA 2022
        self.mu_n = -1.91304276  # Neutron magnetic moment, CODATA 2022
        self.xi = self.mu_p - self.mu_n - 1.0  # Anomalous magnetic moment difference

    def get_kinematic_bounds(self, E_nu):
        """
        Calculates the kinematic bounds for the outgoing positron energy E_e given the incoming neutrino energy E_nu.

        Parameters:
        ---------
        E_nu : float or array-like
            Incoming neutrino energy in MeV.

        Returns:
        ---------
        E_e_min : float or array-like
            Minimum outgoing positron energy in MeV.
        E_e_max : float or array-like
            Maximum outgoing positron energy in MeV.
        """
        # First check if E_nu is above threshold
        E_nu = np.atleast_1d(E_nu)
        E_nu_threshold = ((self.m_n + self.m_e) ** 2 - self.m_p**2) / (2 * self.m_p)
        below_threshold = E_nu < E_nu_threshold

        # Calculate using Eq. 12 from Strumia-Vissani
        # Skip below-threshold values
        E_e_min = np.zeros_like(E_nu)
        E_e_max = np.zeros_like(E_nu)
        E_nu_valid = E_nu[~below_threshold]
        s = self.m_p**2 + 2 * self.m_p * E_nu_valid
        E_nu_cm = (s - self.m_p**2) / (2 * np.sqrt(s))
        E_e_cm = (s - self.m_n**2 + self.m_e**2) / (2 * np.sqrt(s))
        p_e_cm = np.sqrt(E_e_cm**2 - self.m_e**2)
        E_e_min[~below_threshold] = (
            E_nu_valid - self.delta - E_nu_cm * (E_e_cm + p_e_cm) / self.m_p
        )
        E_e_max[~below_threshold] = (
            E_nu_valid - self.delta - E_nu_cm * (E_e_cm - p_e_cm) / self.m_p
        )
        return E_e_min, E_e_max

    def calc_ABC(self, t, f1, f2, g1, g2):
        """
        Calculates A, B, C coefficients using Eq. 6 from Strumia-Vissani.
        Note: The inputs f1, f2, g1, g2 are real.

        Parameters:
        ----------
        t : float or array-like
            Mandelstam variable t (MeV^2).
        f1, f2, g1, g2 : float or array-like
            Form factors at momentum transfer t.

        Returns:
        -------
        A, B, C : float or array-like
            Coefficients for the matrix element calculation.
        """
        me2 = self.m_e**2
        M2 = self.M**2
        Delta2 = self.Delta**2

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

        term3 = -32 * me2 * self.M * self.Delta * g1 * (f1 + f2)

        A = (term1 + term2 + term3) / 16.0

        # Term 16B:
        termB_num = (
            16 * t * g1 * (f1 + f2)
            + (4 * me2 * self.Delta * (f2**2 + f1 * f2 + 2 * g1 * g2)) / self.M
        )
        B = termB_num / 16.0

        # Term 16C:
        termC_num = 4 * (f1**2 + g1**2) - (t * f2**2) / M2
        C = termC_num / 16.0

        return A, B, C

    def differential_cross_section(self, E_nu, E_e):
        """
        Calculates the differential cross section dsigma/dE_e for IBD using the exact Strumia-Vissani formula (Eq. 11).

        Parameters:
        ----------
        E_nu : float
            Incoming neutrino energy in MeV.
        E_e : float
            Outgoing positron energy in MeV.

        Returns:
        -------
        dsigma_dE_corrected : float
            Differential cross section in cm^2/MeV.
        """
        # Kinematic bounds check
        E_nu = np.atleast_1d(E_nu)
        E_e = np.atleast_1d(E_e)
        E_nu, E_e = np.broadcast_arrays(
            E_nu, E_e
        )  # Ensure they can be broadcast together
        E_e_min, E_e_max = self.get_kinematic_bounds(E_nu)
        kinematic_mask = (
            (E_e >= E_e_min)
            & (E_e <= E_e_max)
            & (E_nu >= ((self.m_n + self.m_e) ** 2 - self.m_p**2) / (2 * self.m_p))
        )

        # As usual, we skip out-of-bounds values
        E_nu_valid = E_nu[kinematic_mask]
        E_e_valid = E_e[kinematic_mask]
        dsigma_dE_corrected = np.zeros_like(E_e)
        if E_nu_valid.size == 0:
            return dsigma_dE_corrected

        # Mandelstam variables
        s = self.m_p**2 + 2 * self.m_p * E_nu_valid
        s_minus_u = 2 * self.m_p * (E_nu_valid + E_e_valid) - self.m_e**2
        t = self.m_n**2 - self.m_p**2 - 2 * self.m_p * (E_nu_valid - E_e_valid)

        # Form Factors
        t_div_MV2 = t / self.M_V2
        t_div_MA2 = t / self.M_A2
        DV = 1.0 / (1 - t_div_MV2) ** 2
        DA = 1.0 / (1 - t_div_MA2) ** 2

        tau = t / (4 * self.M**2)
        f1 = DV * (1 - (1 + self.xi) * tau) / (1 - tau)
        f2 = DV * self.xi / (1 - tau)
        g1 = self.g_1_0 * DA
        g2 = 2 * self.M**2 * g1 / (self.m_pi**2 - t)

        # Get Exact A, B, C
        A, B, C = self.calc_ABC(t, f1, f2, g1, g2)

        # Calculate Matrix Element
        M2 = A - s_minus_u * B + s_minus_u**2 * C

        # Differential Cross Section, Eq. 5
        dsigma_dt = (
            (self.G_F**2 * self.cos_theta_c**2)
            / (2 * np.pi * (s - self.m_p**2) ** 2)
            * M2
        )
        # Eq. 11 includes a factor of 2*m_p
        dsigma_dE = 2 * self.m_p * dsigma_dt

        # Radiative Correction
        rad = 1.0 + (self.alpha / np.pi) * (
            6.0
            + 1.5 * np.log(self.m_p / (2 * E_e_valid))
            + 1.2 * (self.m_e / E_e_valid) ** 1.5
        )

        # Sommerfeld Correction is not needed for IBD since the final state has a neutron and a positron
        # eta = 2 * np.pi * self.alpha / np.sqrt(1 - (self.m_e / E_e_valid)**2)
        # sommerfeld = eta / (1 - np.exp(-eta))

        # Finally, transform to cm^2
        dsigma_dE_corrected[kinematic_mask] = dsigma_dE * rad * (self.hbar_c) ** 2
        return dsigma_dE_corrected

    def get_E_dep_pdf(self, E_nu, E_dep):
        """
        Calculates the probability density function (PDF) of deposited energy E_dep for a given incoming neutrino energy E_nu.

        Parameters:
        ----------
        E_nu : float
            Incoming neutrino energy in MeV.
        E_dep : array-like
            Deposited energy array in MeV.

        Returns:
        -------
        pdf : array-like
            Probability density function values corresponding to E_dep.
        """
        E_nu = np.atleast_1d(E_nu)
        E_dep = np.atleast_1d(E_dep)
        # E_nu, E_dep = np.broadcast_arrays(E_nu, E_dep) # Ensure they can be broadcast together
        sigma = self.differential_cross_section(E_nu, E_dep - self.m_e)

        # Normalize by the total cross section
        sigma_total = self.get_total_cross_section(E_nu)

        if sigma_total == 0:
            pdf = np.zeros_like(E_dep)
        else:
            pdf = sigma / sigma_total
        return pdf

    def get_total_cross_section(self, E_nu):
        """
        Calculates the total cross section for IBD by integrating the differential cross section over allowed positron energies.

        Parameters:
        ----------
        E_nu : float or array-like
            Incoming neutrino energy in MeV.

        Returns:
        -------
        total_sigma : float or array-like
            Total cross section in cm^2.
        """
        E_nu = np.atleast_1d(E_nu)
        E_min, E_max = self.get_kinematic_bounds(E_nu)
        # Use scipy.integrate.quad for better accuracy
        total_sigma = np.zeros_like(E_nu)
        for i, E_nu_val in enumerate(np.atleast_1d(E_nu)):
            E_min_val, E_max_val = E_min[i], E_max[i]
            if E_max_val <= E_min_val:
                total_sigma[i] = 0.0
                continue
            total_sigma[i], abs_error = scipy.integrate.quad(
                lambda E_e: self.differential_cross_section(E_nu_val, E_e),
                E_min_val,
                E_max_val,
            )
            if total_sigma[i] > 0 and abs_error / total_sigma[i] > 1e-5:
                print(
                    f"Warning: High relative error in integration: {abs_error / total_sigma[i]:.2e}"
                )
        return total_sigma

    def build_total_cross_section_table(self, E_nu_grid):
        """
        Builds a lookup table for total cross sections over a grid of neutrino energies.

        Parameters:
        ----------
        E_nu_grid : array-like
            Grid of incoming neutrino energies in MeV.

        Returns:
        -------
        sigma_table : array-like
            Total cross section values corresponding to E_nu_grid in cm^2.
        """
        E_nu_grid = np.atleast_1d(E_nu_grid)
        sigma_table = np.zeros_like(E_nu_grid)
        for i, E_nu in enumerate(
            tqdm(E_nu_grid, desc="Building total cross section table")
        ):
            sigma_table[i] = self.get_total_cross_section(E_nu)

        self.sigma_table = sigma_table
        self.E_nu_grid = E_nu_grid

    def get_total_cross_section_from_table(self, E_nu):
        """
        Retrieves total cross section values from the precomputed table using interpolation.

        Parameters:
        ----------
        E_nu : float or array-like
            Incoming neutrino energy in MeV.

        Returns:
        -------
        sigma : float or array-like
            Total cross section in cm^2.
        """
        if not hasattr(self, "sigma_table"):
            raise ValueError(
                "Total cross section table not built. Call build_total_cross_section_table() first."
            )

        E_nu = np.atleast_1d(E_nu)
        sigma = np.interp(E_nu, self.E_nu_grid, self.sigma_table, left=0.0, right=0.0)
        return sigma
