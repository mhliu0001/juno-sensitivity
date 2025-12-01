import numpy as np
import scipy
from neutrino_survival import NeutrinoSurvival
from ibd_kinematics import StrumiaVissani


class NeutrinoFlux:
    def __init__(
        self, neutrino_survival, sv, reactor_data=None, N_p=1.44e33, eff=0.822
    ):
        """
        Initialize the NeutrinoFlux class.

        Parameters:
        ----------
        neutrino_survival : NeutrinoSurvival
            An instance of the NeutrinoSurvival class to calculate survival probabilities.
        sv : StrumiaVissani
            An instance of the StrumiaVissani class to calculate IBD cross sections.
        reactor_data : dict
            A dictionary where keys are reactor names and values are lists of [power_GW, distance_km].
        N_p : float
            Number of free protons in the detector.
        eff : float
            Detection efficiency.
        """
        self.neutrino_survival = neutrino_survival
        # if not isinstance(neutrino_survival, NeutrinoSurvival):
        #     raise ValueError(
        #         "neutrino_survival must be an instance of NeutrinoSurvival class."
        #     )
        self.sv = sv
        if not isinstance(sv, StrumiaVissani):
            raise ValueError("sv must be an instance of StrumiaVissani class.")
        if reactor_data is None:
            self.reactor_data = {
                "Taishan_core1": [4.6, 52.71],
                "Taishan_core2": [4.6, 52.64],
                "Yangjiang_core1": [2.9, 52.74],
                "Yangjiang_core2": [2.9, 52.82],
                "Yangjiang_core3": [2.9, 52.41],
                "Yangjiang_core4": [2.9, 52.49],
                "Yangjiang_core5": [2.9, 52.11],
                "Yangjiang_core6": [2.9, 52.19],
                "DayaBay": [17.4, 215.0],
            }
        else:
            self.reactor_data = reactor_data
        self.N_p = N_p
        self.eff = eff
        self.mueller_coefficients = {
            "U238": [4.833e-1, 1.927e-1, -1.283e-1, -6.762e-3, 2.233e-3, -1.536e-4]
        }
        self.huber_coefficients = {
            "U235": [4.367, -4.577, 2.100, -5.294e-1, 6.186e-2, -2.777e-3],
            "Pu239": [4.757, -5.392, 2.563, -6.596e-1, 7.820e-2, -3.536e-3],
            "Pu241": [2.990, -2.882, 1.278, -3.343e-1, 3.905e-2, -1.754e-3],
        }
        self.huber_spectra = {
            "U235": "./data/U235-anti-neutrino-flux-250keV.dat",
            "Pu239": "./data/Pu239-anti-neutrino-flux-250keV.dat",
            "Pu241": "./data/Pu241-anti-neutrino-flux-250keV.dat",
        }
        self.huber_spectra_skiprows = (
            19  # Number of rows to skip in Huber spectra files
        )
        self.fission_energies = {
            "U235": 202.36,  # in MeV
            "U238": 205.99,
            "Pu239": 211.12,
            "Pu241": 214.26,
        }
        self.fission_fractions = {
            "U235": 0.58,
            "U238": 0.07,
            "Pu239": 0.30,
            "Pu241": 0.05,
        }
        self.daya_bay_scale_file = "./data/daya_bay_flux_ratio.csv"
        self.daya_bay_non_eq_corr_file = "./data/daya_bay_non_eq_corr.csv"
        self.daya_bay_spent_fuel_corr_file = "./data/daya_bay_spent_fuel_corr.csv"

    def get_isotope_flux(self, isotope):
        """
        Get the anti-neutrino flux function for a given isotope.

        Parameters:
        ----------
        isotope : str
            Isotope name, one of 'U235', 'U238', 'Pu239', 'Pu241'.

        Returns:
        -------
        flux_func : function
            A function that takes neutrino energy (MeV) as input and returns the flux (per fission per MeV).
        """
        assert isotope in [
            "U235",
            "U238",
            "Pu239",
            "Pu241",
        ], "Isotope must be one of U235, U238, Pu239, Pu241"
        if isotope == "U238":
            # Use Mueller model
            coefficients = self.mueller_coefficients["U238"]

            def flux_func(E_nu):
                E_nu = np.atleast_1d(E_nu)
                return np.exp(
                    np.sum(
                        [
                            coefficients[i] * E_nu ** (i)
                            for i in range(len(coefficients))
                        ],
                        axis=0,
                    )
                )

            return flux_func
        else:
            # Use Huber model
            spectrum = np.loadtxt(
                self.huber_spectra[isotope], skiprows=self.huber_spectra_skiprows
            )
            flux_interp = scipy.interpolate.interp1d(
                spectrum[:, 0], spectrum[:, 1], kind="linear", fill_value="extrapolate"
            )

            # For energy larger than 8MeV, we extrapolate using the parameters given in the Huber paper
            def flux_func(E_nu):
                E_nu = np.atleast_1d(E_nu)
                flux = flux_interp(E_nu)
                out_of_bounds = (E_nu < spectrum[0, 0]) | (E_nu > spectrum[-1, 0])
                flux[out_of_bounds] = np.exp(
                    np.sum(
                        [
                            self.huber_coefficients[isotope][i]
                            * E_nu[out_of_bounds] ** (i)
                            for i in range(len(self.huber_coefficients[isotope]))
                        ],
                        axis=0,
                    )
                )
                return flux

            return flux_func

    def one_reactor_unoscillated_flux(self, power_GW, distance_km):
        """
        Calculate the unoscillated neutrino flux from one reactor at a given distance.
        This calculation sums up contributions from U235, U238, Pu239, and Pu241, and
        scales the flux according to the Daya Bay scaling factor with respect to the
        Huber-Mueller model.

        Parameters:
        ----------
        power_GW : float
            Reactor thermal power in GW.
        distance_km : float
            Distance from the reactor to the detector in km.

        Returns:
        -------
        flux_func : function
            A function that takes neutrino energy (MeV) as input and returns the flux
            (per MeV per second per m^2).
        """
        # Calculate total fission rate (in fissions per second)
        total_fission_rate = (
            power_GW
            * 1e3
            / scipy.constants.e
            / sum(
                [
                    self.fission_fractions[iso] * self.fission_energies[iso]
                    for iso in self.fission_fractions
                ]
            )
        )

        # Get isotope flux functions (per fission per MeV)
        flux_funcs = {iso: self.get_isotope_flux(iso) for iso in self.fission_fractions}

        # Get scale factor from Daya Bay
        daya_bay_scale_factor = np.loadtxt(self.daya_bay_scale_file, delimiter=",")
        daya_bay_non_eq_corr = np.loadtxt(self.daya_bay_non_eq_corr_file, delimiter=",")
        daya_bay_spent_fuel_corr = np.loadtxt(
            self.daya_bay_spent_fuel_corr_file, delimiter=","
        )

        def get_scale_factor(E_nu):
            # Note that this returns boundary values for out-of-bounds energies
            return (
                np.interp(
                    E_nu, daya_bay_scale_factor[:, 0], daya_bay_scale_factor[:, 1]
                )
                * (
                    1
                    + np.interp(
                        E_nu, daya_bay_non_eq_corr[:, 0], daya_bay_non_eq_corr[:, 1]
                    )
                )
                * (
                    1
                    + np.interp(
                        E_nu,
                        daya_bay_spent_fuel_corr[:, 0],
                        daya_bay_spent_fuel_corr[:, 1],
                    )
                )
            )

        def flux_func(E_nu):
            flux = np.zeros_like(E_nu)
            for iso in self.fission_fractions:
                flux += self.fission_fractions[iso] * flux_funcs[iso](E_nu)
            # Convert to per m^2 per second at given distance
            flux *= (
                total_fission_rate
                * get_scale_factor(E_nu)
                / (4 * np.pi * (distance_km * 1e3) ** 2)
            )
            return flux

        return flux_func

    def all_reactors_unoscillated_flux(self, reactor_data):
        """
        Calculate the total unoscillated neutrino flux from all reactors listed in the data file.

        Parameters:
        ----------
        reactor_data : dict
            A dictionary where keys are reactor names and values are lists of [power_GW, distance_km].

        Returns:
        -------
        flux_func : function
            A function that takes neutrino energy (MeV) as input and returns the total flux
            (per MeV per second per m^2).
        """

        def total_flux_func(E_nu):
            total_flux = np.zeros_like(E_nu)
            for power, distance in reactor_data.values():
                total_flux += self.one_reactor_unoscillated_flux(power, distance)(E_nu)
            return total_flux

        return total_flux_func

    def all_reactors_oscillated_flux(self, reactor_data, hierarchy="normal"):
        """
        Calculate the total oscillated neutrino flux from all reactors listed in the data file.

        Parameters:
        ----------
        reactor_data : dict
            A dictionary where keys are reactor names and values are lists of [power_GW, distance_km].
        hierarchy : str


        Returns:
        -------
        flux_func : function
            A function that takes neutrino energy (MeV) as input and returns the total flux
            (per MeV per second per m^2).
        """

        def total_flux_func(E_nu):
            total_flux = np.zeros_like(E_nu)
            for power, distance in reactor_data.values():
                total_flux += self.one_reactor_unoscillated_flux(power, distance)(
                    E_nu
                ) * self.neutrino_survival.survival_probability(
                    E_nu, distance, hierarchy=hierarchy
                )
            return total_flux

        return total_flux_func

    def expected_ibd_event_rate(self, flux_func):
        """
        Calculate the expected IBD event rate in JUNO detector given the neutrino flux function.

        Parameters:
        ----------
        flux_func : function
            A function that takes neutrino energy (MeV) as input and returns the flux
            (per MeV per second per m^2).

        Returns:
        -------
        event_rate : float
            The expected IBD event rate in the JUNO detector (events per second).
        """
        sv = self.sv
        if not hasattr(sv, "sigma_table"):
            print(
                "Warning: StrumiaVissani instance does not have precomputed sigma_table. This may slow down the calculation."
            )

            def rate(E_nu):
                E_nu = np.atleast_1d(E_nu)
                return (
                    flux_func(E_nu)
                    * sv.get_total_cross_section(E_nu)
                    * 1e-4
                    * self.eff
                    * self.N_p
                )  # Convert cm^2 to m^2

            return rate

        else:

            def rate(E_nu):
                E_nu = np.atleast_1d(E_nu)
                return (
                    flux_func(E_nu)
                    * sv.get_total_cross_section_from_table(E_nu)
                    * 1e-4
                    * self.eff
                    * self.N_p
                )  # Convert cm^2 to m^2

            return rate

    def expected_signal_rate(self, E_nu, hierarchy="normal"):
        """
        Calculate the expected IBD signal rate in JUNO detector, given nominal neutrino flux.

        Parameters:
        ----------
        E_nu : array-like
            Neutrino energy in MeV.
        hierarchy : str
            'normal' or 'inverted' mass hierarchy.

        Returns:
        -------
        event_rate : array-like
            The expected IBD event rate in the JUNO detector (events per second).
        """
        flux_func = self.all_reactors_oscillated_flux(
            self.reactor_data, hierarchy=hierarchy
        )
        rate_func = self.expected_ibd_event_rate(flux_func)
        return rate_func(E_nu)
