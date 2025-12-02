import numpy as np
import scipy

from neutrino_survival import NeutrinoSurvival
from ibd_kinematics import StrumiaVissani
from detector_response import JUNODetector
from neutrino_flux import NeutrinoFlux


class JUNOSignalAndBackground:
    def __init__(
        self,
        ns,
        sv,
        detector,
        nf,
        exposure_years=6.5,
        bin_edges=None,
        background_rate_file="./data/juno_background.csv",
    ):
        """Class to calculate expected signal and background events in JUNO.

        Parameters:
        ----------
        ns : NeutrinoSurvival
            Instance of NeutrinoSurvival class.
        sv : StrumiaVissani
            Instance of StrumiaVissani class.
        detector : JUNODetector
            Instance of JUNODetector class.
        nf : NeutrinoFlux
            Instance of NeutrinoFlux class.
        exposure_years : float
            Exposure time in years.
        bin_edges : np.ndarray
            Edges of reconstructed energy bins in MeV.
        background_rate_file : str
            Path to CSV file containing background rate data."""
        self.ns = ns
        self.sv = sv
        self.detector = detector
        self.nf = nf
        self.exposure_years = exposure_years
        self.background_rate_file = background_rate_file
        if bin_edges is None:
            self.bin_edges = (
                [
                    0.8,
                ]
                + list(np.arange(0.94, 7.43, 0.02))
                + list(np.arange(7.44, 7.79, 0.04))
                + list(np.arange(7.80, 8.21, 0.10))
                + [12.0]
            )
            self.bin_edges = np.array(self.bin_edges)  # MeV
        else:
            self.bin_edges = np.array(bin_edges)

        self.build_background()

    def build_background(self):
        """Load and prepare background event counts per bin."""
        background_rate_file = self.background_rate_file

        # Load background spectrum
        background_rate_data = np.loadtxt(background_rate_file, delimiter=",")
        background_rate_energies = background_rate_data[:, 0]  # MeV
        background_rate = background_rate_data[:, 1]  # events per 20 keV per 6.5 years

        # Convert to events / MeV / s
        background_rate /= 6.5 * 365.25 * 24 * 3600 * 20e-3

        # Global normalization correction to 4.11 events/day between 0.8 and 12 MeV
        e_norm_min, e_norm_max = 0.8, 12.0

        # Restrict to the normalization window
        norm_mask = (background_rate_energies >= e_norm_min) & (
            background_rate_energies <= e_norm_max
        )
        E_norm = background_rate_energies[norm_mask]
        rate_norm = background_rate[norm_mask]

        # Integral over [0.8, 12] using trapezoidal rule (events/s)
        raw_int = np.trapz(rate_norm, E_norm)
        raw_bkg_rate_per_day = raw_int * 86400.0  # events/day

        # We subtract a constant rate "correction" so that the total becomes 4.11/day:
        # ∫ (rate - correction) dE = 4.11 / 86400
        correction = (raw_bkg_rate_per_day - 4.11) / (
            (e_norm_max - e_norm_min) * 86400.0
        )

        # Apply correction on the grid
        background_rate_corrected = background_rate - correction

        # For diagnostic printout, approximate the original rate at 1.5 MeV
        rate_at_1p5 = np.interp(1.5, background_rate_energies, background_rate)
        # print(f"Background rate correction: {correction / rate_at_1p5} at 1.5 MeV")

        # Fast binning using cumulative integral
        # cumulative_trapezoid gives integral from E_min to each energy point
        cum_int = scipy.integrate.cumulative_trapezoid(
            background_rate_corrected, background_rate_energies, initial=0.0
        )  # units: events/s as a function of upper limit

        # Interpolate this cumulative integral at all bin edges
        # (clip to the tabulated energy range to avoid extrapolation issues)
        e_min_tab, e_max_tab = background_rate_energies[0], background_rate_energies[-1]
        edges_clipped = np.clip(self.bin_edges, e_min_tab, e_max_tab)
        cum_at_edges = np.interp(edges_clipped, background_rate_energies, cum_int)

        # Events per second in each bin = F(E_high) - F(E_low)
        bin_counts_background_per_sec = np.diff(cum_at_edges)

        # Normalize to data taking time
        exposure_time = self.exposure_years * 86400.0 * 365.25  # seconds
        bin_counts_background = bin_counts_background_per_sec * exposure_time

        self.bin_counts_background = bin_counts_background

    def calculate_expected_events(self, hierarchy="normal"):
        """Calculate expected event rates for given hierarchy and exposure time.

        Parameters:
        ----------
        hierarchy : str
            'normal' or 'inverted'.

        Returns:
        -------
        bin_counts : np.ndarray
            Expected event counts per reconstructed energy bin.
        """
        # Ensure response matrix and cross section table are loaded
        assert self.detector.R_matrix_prepared, "Detector response matrix not loaded."
        assert hasattr(self.sv, "sigma_table"), "Cross section table not built."

        # --- Signal in true E_nu ---
        E_nu_grid = self.detector.E_nu_grid  # MeV
        E_rec_grid = self.detector.E_rec_grid  # MeV (must be sorted ascending)
        signal_rate = self.nf.expected_signal_rate(
            E_nu_grid, hierarchy
        )  # per MeV per second

        # Convolve with detector response → rate vs reconstructed energy
        signal_rate_rec = self.detector.apply_response_matrix(
            signal_rate, E_nu_grid, E_rec_grid
        )  # per MeV per second

        # Scale by exposure time (seconds)
        exposure_time = self.exposure_years * 86400.0 * 365.25
        signal_rate_rec *= exposure_time  # → total expected events per MeV

        # --- Fast binning using cumulative integral ---
        # cumulative_trapezoid gives integral from E_rec_grid[0] to each grid point
        # I_k = ∫_{E0}^{E_k} signal_rate_rec(E) dE
        cum_int = scipy.integrate.cumulative_trapezoid(
            signal_rate_rec, E_rec_grid, initial=0.0
        )

        # Interpolate this cumulative integral at the bin edges
        # (vectorized, much faster than quad in a Python loop)
        cum_at_edges = np.interp(self.bin_edges, E_rec_grid, cum_int)

        # Integral in each bin = F(E_high) - F(E_low)
        bin_counts_signal = np.diff(cum_at_edges)

        # Add precomputed background
        bin_counts = bin_counts_signal + self.bin_counts_background

        return bin_counts
