import struct
import numpy as np
import scipy
from ibd_kinematics import StrumiaVissani
from tqdm import tqdm
import hashlib


class JUNODetector:
    def __init__(
        self,
        strumia_vissani,
        nonlinearity_file="./data/non_linear_response.csv",
        a=0.0261,
        b=0.0064,
        c=0.0120,
    ):
        """Initialize JUNO Detector response with nonlinearity and resolution parameters.

        Parameters
        ----------
        strumia_vissani : StrumiaVissani
            An instance of the StrumiaVissani class for IBD kinematics.
        nonlinearity_file : str
            Path to CSV file containing nonlinearity curve data.
        a : float
            Stochastic term coefficient for energy resolution.
        b : float
            Constant term coefficient for energy resolution.
        c : float
            Noise term coefficient for energy resolution.
        """
        if not isinstance(strumia_vissani, StrumiaVissani):
            raise ValueError(
                "strumia_vissani must be an instance of StrumiaVissani class."
            )
        self.strumia_vissani = strumia_vissani

        # Load Nonlinearity
        # Expected columns: E_dep, ratio (E_vis/E_dep)
        nonlinear_curve_file = np.loadtxt(nonlinearity_file, delimiter=",")
        e_dep_data = nonlinear_curve_file[:, 0]
        e_vis_data = nonlinear_curve_file[:, 1] * e_dep_data

        # Build polynomial fits
        self.coeffs_fwd = np.polyfit(e_dep_data, e_vis_data, deg=5)  # E_dep to E_vis
        self.coeffs_inv = np.polyfit(e_vis_data, e_dep_data, deg=5)  # E_vis to E_dep
        self.spline_vis_to_dep = np.poly1d(self.coeffs_inv)
        self.spline_dep_to_vis = np.poly1d(self.coeffs_fwd)
        self.deriv_vis_to_dep = self.spline_vis_to_dep.deriv()

        # Resolution Parameters (JUNO Paper)
        # sigma/E = sqrt( a^2/E + b^2 + c^2/E^2 )
        self.a = a  # 2.61%
        self.b = b  # 0.64%
        self.c = c  # 1.20% MeV (Noise term)

        self.R_matrix_prepared = False

    def get_E_vis_pdf(self, E_dep_pdf):
        """
        Transform f(E_dep) to f(E_vis) using the nonlinearity curve.

        Parameters
        ----------
        E_dep_pdf : function
            A function that takes E_nu and E_dep (MeV) and returns f(E_dep|E_nu).

        Returns
        -------
        E_vis_pdf : function
            A function that takes E_nu and E_vis (MeV) and returns f(E_vis|E_nu).
        """

        def E_vis_pdf(E_nu, E_vis):
            E_dep = self.spline_vis_to_dep(E_vis)
            jacobian = np.abs(self.deriv_vis_to_dep(E_vis))
            return E_dep_pdf(E_nu, E_dep) * jacobian

        return E_vis_pdf

    def get_resolution(self, E_vis):
        """Calculate sigma (MeV) for a given E_vis (MeV).

        Parameters
        ----------
        E_vis : float or np.ndarray
            Visible energy in MeV.

        Returns
        -------
        sigma : float or np.ndarray
            Energy resolution sigma in MeV.
        """
        # Formula: sigma = E * sqrt( (a/sqrt E)^2 + b^2 + (c/E)^2 )
        #                = sqrt( a^2 E + b^2 E^2 + c^2 )
        sigma = np.sqrt((self.a**2 * E_vis) + (self.b * E_vis) ** 2 + self.c**2)
        return sigma

    def get_E_rec_pdf(self, E_dep_pdf, E_dep_min, E_dep_max):
        """
        Construct f(E_rec|E_nu) by convolving f(E_vis|E_nu) with detector resolution.
        Uses numerical integration (quad) for each E_rec point and is slow. We recommend
        using the matrix method `get_E_rec_pdf_grid` for efficiency.

        Parameters
        ----------
        E_dep_pdf : function
            A function that takes E_dep (MeV) and returns f(E_dep|E_nu).

        Returns
        -------
        E_rec_pdf : function
            A function that takes E_rec (MeV) and returns f(E_rec|E_nu).
        """

        def E_rec_pdf(E_nu, E_rec):
            # Convolution integral
            def integrand(E_vis):
                sigma = self.get_resolution(E_vis)
                gauss = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
                    -0.5 * ((E_rec - E_vis) / sigma) ** 2
                )
                return self.get_E_vis_pdf(E_dep_pdf)(E_nu, E_vis) * gauss

            E_vis_min = E_rec - 5 * self.get_resolution(E_rec)
            E_vis_max = E_rec + 5 * self.get_resolution(E_rec)
            # We set practical limits based on E_dep range
            E_vis_min = max(E_vis_min, self.spline_dep_to_vis(E_dep_min))
            E_vis_max = min(E_vis_max, self.spline_dep_to_vis(E_dep_max))

            integral, abs_error = scipy.integrate.quad(integrand, E_vis_min, E_vis_max)
            if integral > 0 and abs_error / integral > 1e-7:
                print(
                    f"Warning: High relative error in convolution integral at E_rec={E_rec} MeV: {abs_error/integral:.2e}"
                )
            return integral

        return E_rec_pdf

    def get_E_rec_pdf_grid(self, E_vis_grid, E_rec_grid, pdf_vis_grid):
        """
        Convolves f(E_vis) with resolution using a matrix multiplication method.
        This replaces the slow 'quad' integration loop.

        Parameters
        ----------
        E_vis_grid : np.ndarray
            1D array of visible energy grid points (MeV).
        E_rec_grid : np.ndarray
            1D array of reconstructed energy grid points (MeV).
        pdf_vis_grid : np.ndarray
            1D array of f(E_vis) values corresponding to E_vis_grid.

        Returns
        -------
        pdf_rec_grid : np.ndarray
            1D array of f(E_rec) values corresponding to E_rec_grid.
        """
        # Create meshgrids for broadcasting: Rows=Rec, Cols=Vis
        # If E_vis_grid has shape (N_vis,), E_rec_grid has shape (N_rec,)
        # then Vis_2D and Rec_2D will have shape (N_rec, N_vis)
        Vis_2D, Rec_2D = np.meshgrid(E_vis_grid, E_rec_grid)

        # Calculate Sigma for every Visible Energy column
        Sigmas = self.get_resolution(Vis_2D)

        # Gaussian Kernel
        # P(E_rec | E_vis)
        Gauss_Matrix = (1.0 / (np.sqrt(2 * np.pi) * Sigmas)) * np.exp(
            -0.5 * ((Rec_2D - Vis_2D) / Sigmas) ** 2
        )

        # Apply Simpson's Rule weights for integration over E_vis
        simpson_weights = np.ones_like(E_vis_grid)
        simpson_weights[1:-1:2] = 4
        simpson_weights[2:-2:2] = 2
        simpson_weights *= np.gradient(E_vis_grid) / 3.0
        Gauss_Matrix *= simpson_weights[np.newaxis, :]

        # The Convolution
        pdf_rec_grid = np.dot(Gauss_Matrix, pdf_vis_grid)

        return pdf_rec_grid

    def build_response_matrix(self, E_nu_grid, E_rec_grid, bins_E_vis=1001, eps=1e-3):
        """
        Build the detector response matrix R(E_rec, E_nu) for a given neutrino energy E_nu.
        Each column corresponds to f(E_rec | E_nu_i) for fixed E_nu_i.

        Parameters
        ----------
        E_nu_grid : np.ndarray
            1D array of neutrino energy grid points (MeV).
        E_rec_grid : np.ndarray
            1D array of reconstructed energy grid points (MeV).
        bins_E_vis : int, optional
            Number of bins for the visible energy grid used in the convolution (default is 1001).
        eps : float, optional
            Small margin added to E_vis bounds to avoid edge issues (default is 1e-3).

        Returns
        -------
        R_matrix : np.ndarray
            2D array of shape (len(E_rec_grid), len(E_nu_grid)) representing the response matrix.
        """
        R_matrix = np.zeros((len(E_rec_grid), len(E_nu_grid)))

        for i, E_nu in enumerate(
            tqdm(E_nu_grid, desc="Building Detector Response Matrix")
        ):
            # Calculate E_vis bounds for this E_nu
            # This is necessary to limit the integration range
            E_e_min, E_e_max = self.strumia_vissani.get_kinematic_bounds(E_nu)
            E_dep_min = E_e_min[0] + self.strumia_vissani.m_e
            E_dep_max = E_e_max[0] + self.strumia_vissani.m_e
            # We add an eps margin to avoid edge issues from the polynomial fit
            E_vis_min = self.spline_dep_to_vis(E_dep_min) - eps
            E_vis_max = self.spline_dep_to_vis(E_dep_max) + eps
            assert E_vis_max >= E_vis_min, "E_vis_max must be greater than E_vis_min"
            # We compute the matrix-smearing result
            E_vis_grid = np.linspace(E_vis_min, E_vis_max, bins_E_vis)  # Grid for E_vis

            pdf_vis_grid = self.get_E_vis_pdf(self.strumia_vissani.get_E_dep_pdf)(
                E_nu, E_vis_grid
            )
            pdf_rec_grid = self.get_E_rec_pdf_grid(E_vis_grid, E_rec_grid, pdf_vis_grid)

            R_matrix[:, i] = pdf_rec_grid

        self.R_matrix = R_matrix
        self.E_nu_grid = E_nu_grid
        self.E_rec_grid = E_rec_grid
        self.R_matrix_prepared = True
        return R_matrix

    def stable_hash(self):
        """
        Generate a stable hash based on the detector parameters a, b, c.

        Returns
        -------
        hash_value : int
            An integer hash value.
        """
        # Pack the float parameters into bytes
        data = struct.pack("ddd", self.a, self.b, self.c)
        return int(hashlib.blake2b(data, digest_size=8).hexdigest(), 16) % (10**8)

    def save_response_matrix(self):
        """
        Save the built response matrix to a compressed .npz file.
        """
        if not hasattr(self, "R_matrix_prepared") or not self.R_matrix_prepared:
            raise RuntimeError(
                "Response matrix not built. Call build_response_matrix() first."
            )

        # We need to calculate a hash for the current parameters
        # Now only a, b and c are considered for the response matrix
        filename = f"juno_detector_response_matrix_{self.stable_hash()}.npz"
        np.savez_compressed(
            filename,
            E_nu_grid=self.E_nu_grid,
            E_rec_grid=self.E_rec_grid,
            f_Erec_Enu=self.R_matrix,
        )
        print(f"Response matrix saved to {filename}")

    def load_response_matrix(self):
        """
        Load a response matrix from a compressed .npz file.

        Parameters
        ----------
        filename : str
            Path to the .npz file containing the response matrix data.
        """
        filename = f"juno_detector_response_matrix_{self.stable_hash()}.npz"
        data = np.load(filename)
        self.E_nu_grid = data["E_nu_grid"]
        self.E_rec_grid = data["E_rec_grid"]
        self.R_matrix = data["f_Erec_Enu"]
        self.R_matrix_prepared = True
        print(f"Response matrix loaded from {filename}")

    def apply_response_matrix(self, flux_Enu, E_nu_grid, E_rec_grid):
        """
        Apply the detector response matrix to an input neutrino flux to get the reconstructed energy spectrum.

        Parameters
        ----------
        flux_Enu : np.ndarray
            1D array of neutrino flux values corresponding to E_nu_grid.
        E_nu_grid : np.ndarray
            1D array of neutrino energy grid points (MeV).
        E_rec_grid : np.ndarray
            1D array of reconstructed energy grid points (MeV).

        Returns
        -------
        spectrum_Erec : np.ndarray
            1D array of reconstructed energy spectrum values corresponding to E_rec_grid.
        """
        if not hasattr(self, "R_matrix_prepared") or not self.R_matrix_prepared:
            raise RuntimeError(
                "Response matrix not built. Call build_response_matrix() first."
            )
        if not np.array_equal(E_nu_grid, self.E_nu_grid):
            raise ValueError(
                "Input E_nu_grid does not match the one used to build the response matrix."
            )
        if not np.array_equal(E_rec_grid, self.E_rec_grid):
            raise ValueError(
                "Input E_rec_grid does not match the one used to build the response matrix."
            )

        # Matrix multiplication to get f(E_rec)
        # We also use simpson's rule for integration over E_nu
        simpson_weights = np.ones_like(E_nu_grid)
        simpson_weights[1:-1:2] = 4
        simpson_weights[2:-2:2] = 2
        simpson_weights *= np.gradient(E_nu_grid) / 3.0
        spectrum_Erec = np.dot(self.R_matrix, flux_Enu * simpson_weights)
        return spectrum_Erec
