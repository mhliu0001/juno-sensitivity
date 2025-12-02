import numpy as np
import scipy
from neutrino_survival import NeutrinoSurvivalDMP, NeutrinoSurvival
from ibd_kinematics import StrumiaVissani
from detector_response import JUNODetector
from neutrino_flux import NeutrinoFlux
from signal_and_background import JUNOSignalAndBackground


class JUNOSensitivity:
    def __init__(self, sv, detector, use_dmp=True):
        """
        Initialize JUNOSensitivity with StrumiaVissani and JUNODetector instances.

        Parameters:
        ----------
        sv : StrumiaVissani
            Instance of StrumiaVissani for cross section calculations.
        detector : JUNODetector
            Instance of JUNODetector for detector response.
        use_dmp : bool, optional
            Whether to use NeutrinoSurvivalDMP for matter effects. Default is True.
        """
        self.sv = sv
        self.detector = detector
        self.use_dmp = use_dmp
        # Check that cross section and response matrices are built
        if not hasattr(self.sv, "sigma_table"):
            raise ValueError("StrumiaVissani cross section table not built.")
        if not self.detector.R_matrix_prepared:
            raise ValueError("JUNO detector response matrix not built.")

    def get_signal_and_bkg(
        self,
        delta_m21_squared,
        delta_m32_squared,
        s_theta_12_squared,
        s_theta_13_squared,
        exposure_years=6.5,
        ns_kwargs=None,
    ):
        """
        Calculate expected signal and background event counts.

        Parameters:
        ----------
        delta_m21_squared : float
            Solar mass-squared difference in eV^2.
        delta_m32_squared : float
            mass-squared difference of m_3 and m_2 in eV^2. Positive for normal hierarchy, negative for inverted.
        s_theta_12_squared : float
            Sine squared of theta_12 mixing angle.
        s_theta_13_squared : float
            Sine squared of theta_13 mixing angle.
        exposure_years : float, optional
            Exposure time in years. Default is 6.5 years.
        ns_kwargs : dict, optional
            Additional keyword arguments for NeutrinoSurvival initialization.

        Returns:
        -------
        bin_counts : array-like
            Expected event counts in each energy bin.
        """
        # Create NeutrinoSurvival and NeutrinoFlux instances with given parameters
        if ns_kwargs is None:
            ns_kwargs = {}
        if delta_m32_squared > 0:
            hierarchy = "normal"
            delta_m32_squared_input = [delta_m32_squared, -2.546e-3]
        else:
            hierarchy = "inverted"
            delta_m32_squared_input = [2.453e-3, delta_m32_squared]
        if not self.use_dmp:
            ns = NeutrinoSurvival(
                delta_m21_squared=delta_m21_squared,
                delta_m32_squared=delta_m32_squared_input,
                s_theta_12_squared=s_theta_12_squared,
                s_theta_13_squared=s_theta_13_squared,
                **ns_kwargs,
            )
        else:
            ns = NeutrinoSurvivalDMP(
                delta_m21_squared=delta_m21_squared,
                delta_m32_squared=delta_m32_squared_input,
                s_theta_12_squared=s_theta_12_squared,
                s_theta_13_squared=s_theta_13_squared,
                **ns_kwargs,
            )
        nf = NeutrinoFlux(ns, self.sv)
        # Create JUNOSignalAndBackground instance and calculate expected events
        signal_and_bkg = JUNOSignalAndBackground(
            ns, self.sv, self.detector, nf, exposure_years=exposure_years
        )
        bin_counts = signal_and_bkg.calculate_expected_events(hierarchy=hierarchy)
        return bin_counts

    def build_cost_function(
        self,
        data=None,
        hierarchy="normal",
        exposure_years=6.5,
        ns_kwargs=None,
        gaussian_prior=None,
    ):
        """
        Build a cost function for sensitivity analysis.

        Parameters:
        ----------
        data : array-like, optional
            Observed data to compare against. If None, use expected events.
        hierarchy : str, optional
            Neutrino mass hierarchy, either "normal" or "inverted".
        exposure_years : float, optional
            Exposure time in years. Default is 6.5 years.
        ns_kwargs : dict, optional
            Additional keyword arguments for NeutrinoSurvival initialization.
        gaussian_prior : dict, optional
            Gaussian prior information for parameters. If None, prior is applied to sine squared theta_13 (0.0218 \pm 0.0007).
            If given, should be a dictionary with keys as parameter names and values as (mean, sigma) tuples.

        Returns:
        cost_function : callable
            A function that computes the cost given model parameters.
        """
        if data is None:
            data = self.get_signal_and_bkg(
                delta_m21_squared=7.53e-5,
                delta_m32_squared=2.453e-3 if hierarchy == "normal" else -2.546e-3,
                s_theta_12_squared=0.307,
                s_theta_13_squared=0.0218,
                exposure_years=exposure_years,
                ns_kwargs=ns_kwargs,
            )

        def cost_function(
            delta_m21_squared, delta_m32_squared, s_theta_12_squared, s_theta_13_squared
        ):
            model_counts = self.get_signal_and_bkg(
                delta_m21_squared,
                delta_m32_squared,
                s_theta_12_squared,
                s_theta_13_squared,
                exposure_years=exposure_years,
                ns_kwargs=ns_kwargs,
            )
            # Compute chi-squared or likelihood
            chi2 = np.sum((data - model_counts) ** 2 / model_counts)

            # We also add a Gaussian prior if provided
            if gaussian_prior is not None:
                for param_name, (mean, sigma) in gaussian_prior.items():
                    if param_name == "delta_m21_squared":
                        chi2 += ((delta_m21_squared - mean) / sigma) ** 2
                    elif param_name == "delta_m32_squared":
                        chi2 += ((delta_m32_squared - mean) / sigma) ** 2
                    elif param_name == "s_theta_12_squared":
                        chi2 += ((s_theta_12_squared - mean) / sigma) ** 2
                    elif param_name == "s_theta_13_squared":
                        chi2 += ((s_theta_13_squared - mean) / sigma) ** 2
                    else:
                        raise ValueError(
                            f"Unknown parameter name in prior: {param_name}"
                        )
            else:
                # Default prior on s_theta_13_squared
                chi2 += ((s_theta_13_squared - 0.0218) / 0.0007) ** 2
            return chi2

        return cost_function
