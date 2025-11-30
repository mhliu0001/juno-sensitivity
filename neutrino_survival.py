import numpy as np
import scipy


class NeutrinoSurvival:
    """Class to calculate neutrino survival probabilities in matter."""

    def __init__(
        self,
        s_theta_12_squared=0.307,
        s_theta_13_squared=0.0218,
        s_theta_23_squared=[0.545, 0.547],
        delta_m21_squared=7.53e-5,
        delta_m32_squared=[2.453e-3, -2.546e-3],
        Y_e=0.5,
        rho=2.45,
    ):
        # Neutrino oscillation parameters (PDG 2020)
        self.s_theta_12_squared = s_theta_12_squared
        self.s_theta_13_squared = s_theta_13_squared
        self.s_theta_23_squared = s_theta_23_squared  # normal and inverted hierarchy
        self.delta_cp = 1.36  # radians
        self.delta_m21_squared = delta_m21_squared
        self.delta_m32_squared = delta_m32_squared
        self.G_F = 1.1663787e-5  # GeV^-2
        self.Y_e = Y_e  # electron fraction in matter
        self.rho = rho  # g/cm^3, average Earth crust density
        self.N_A = scipy.constants.Avogadro  # Avogadro's number
        self.hbar_c = (
            scipy.constants.hbar * scipy.constants.c / scipy.constants.e * 1e-7
        )  # GeV*cm

    def matter_potential(self, E_nu):
        """Calculate the matter potential A for neutrinos in matter.

        Parameters:
        ----------
        E_nu : float
            Neutrino energy in MeV.

        Returns:
        -------
        A : float
            Matter potential in eV^2.
        """
        N_e = self.Y_e * self.rho * self.N_A  # electron number density in cm^-3
        N_e_natural = N_e * (self.hbar_c) ** 3  # convert to natural units (GeV^3)
        A = 2 * np.sqrt(2) * self.G_F * N_e_natural * E_nu * 1e-3 * 1e18  # eV^2
        return A

    def effective_parameters(self, E_nu, neutrino="antineutrino", hierarchy="normal"):
        """Calculate the effective mixing angles and mass-squared differences in matter.

        Parameters:
        ----------
        E_nu : float
            Neutrino energy in MeV.
        neutrino : str
            'neutrino' or 'antineutrino'.
        hierarchy : str
            'normal' or 'inverted' mass hierarchy.

        Returns:
        -------
        s_theta_12_squared_m : float
            Effective sin^2(theta_12) in matter.
        s_theta_13_squared_m : float
            Effective sin^2(theta_13) in matter.
        delta_m21_squared_m : float
            Effective delta m^2_21 in matter (eV^2).
        delta_m31_squared_m : float
            Effective delta m^2_31 in matter (eV^2).
        """
        A = self.matter_potential(E_nu)
        delta_m31_squared = (
            self.delta_m32_squared[0] + self.delta_m21_squared
            if hierarchy == "normal"
            else self.delta_m32_squared[1] + self.delta_m21_squared
        )
        assert neutrino in [
            "neutrino",
            "antineutrino",
        ], "neutrino must be 'neutrino' or 'antineutrino'"
        if neutrino == "antineutrino":
            # For antineutrinos, the matter potential changes sign
            A = -A
        A_tilde_13 = (
            A / delta_m31_squared
        )  # dimensionless matter potential for 1-3 sector
        A_tilde_12 = (
            A * (1 - self.s_theta_13_squared) / self.delta_m21_squared
        )  # dimensionless matter potential for 1-2 sector

        # Calculate effective parameters, using approximate analytical formulas
        # Full calculation would involve diagonalizing the Hamiltonian in matter
        c_double_theta_13_m = (1 - 2 * self.s_theta_13_squared) / np.sqrt(
            A_tilde_13**2 - 2 * A_tilde_13 * (1 - 2 * self.s_theta_13_squared) + 1
        )
        s_theta_13_squared_m = 0.5 * (1 - c_double_theta_13_m)
        delta_m31_squared_m = delta_m31_squared * np.sqrt(
            A_tilde_13**2 - 2 * A_tilde_13 * (1 - 2 * self.s_theta_13_squared) + 1
        )

        c_double_theta_12_m = (1 - 2 * self.s_theta_12_squared) / np.sqrt(
            A_tilde_12**2 - 2 * A_tilde_12 * (1 - 2 * self.s_theta_12_squared) + 1
        )
        s_theta_12_squared_m = 0.5 * (1 - c_double_theta_12_m)
        delta_m21_squared_m = self.delta_m21_squared * np.sqrt(
            A_tilde_12**2 - 2 * A_tilde_12 * (1 - 2 * self.s_theta_12_squared) + 1
        )

        return (
            s_theta_12_squared_m,
            s_theta_13_squared_m,
            delta_m21_squared_m,
            delta_m31_squared_m,
        )

    def survival_probability(self, E_nu, L, hierarchy="normal"):
        """Calculate the electron antineutrino survival probability in matter.

        Parameters:
        ----------
        E_nu : float
            Neutrino energy in MeV.
        L : float
            Baseline distance in km.
        hierarchy : str
            'normal' or 'inverted' mass hierarchy.

        Returns:
        -------
        P_ee : float
            Electron antineutrino survival probability.
        """
        # Effective parameters in matter
        (
            s_theta_12_squared_m,
            s_theta_13_squared_m,
            delta_m21_squared_m,
            delta_m31_squared_m,
        ) = self.effective_parameters(
            E_nu, neutrino="antineutrino", hierarchy=hierarchy
        )
        c_theta_12_squared_m = 1 - s_theta_12_squared_m
        c_theta_13_squared_m = 1 - s_theta_13_squared_m
        s_double_theta_12_squared_m = (
            4 * s_theta_12_squared_m * (1 - s_theta_12_squared_m)
        )
        s_double_theta_13_squared_m = (
            4 * s_theta_13_squared_m * (1 - s_theta_13_squared_m)
        )
        delta_m32_squared_m = delta_m31_squared_m - delta_m21_squared_m

        # Calculate oscillation phases
        delta_21 = (
            delta_m21_squared_m * L / (4 * E_nu) / self.hbar_c * 1e-10
        )  # in radians
        delta_31 = (
            delta_m31_squared_m * L / (4 * E_nu) / self.hbar_c * 1e-10
        )  # in radians
        delta_32 = (
            delta_m32_squared_m * L / (4 * E_nu) / self.hbar_c * 1e-10
        )  # in radians

        # Survival probability formula
        P_ee = (
            1
            - s_double_theta_12_squared_m
            * c_theta_13_squared_m**2
            * np.sin(delta_21) ** 2
            - s_double_theta_13_squared_m
            * (
                c_theta_12_squared_m * np.sin(delta_31) ** 2
                + s_theta_12_squared_m * np.sin(delta_32) ** 2
            )
        )

        return P_ee
