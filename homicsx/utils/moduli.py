import numpy as np


def E_nu_to_kappa_mu_3d(
        E: float, 
        nu: float,
):
    """
    Converts Young's modulus (E) and Poisson's ratio (nu) to shear modulus (mu) and bulk modulus (kappa).

    Parameters:
        E (float): Young's modulus.
        nu (float): Poisson's ratio.
    
    Returns:
        mu (float): Shear modulus.
        kappa (float): Bulk modulus.
    """
    mu = E / (2 * (1 + nu))  # Shear modulus (mu)
    kappa = E / (3 * (1 - 2 * nu))  # Bulk modulus (kappa)
    return kappa, mu


def kappa_mu_to_E_nu_3d(
        kappa: float, 
        mu: float,
):
    """
    Converts bulk modulus (kappa) and shear modulus (mu) to Young's modulus (E) and Poisson's ratio (nu).

    Parameters:
        kappa (float): Bulk modulus.
        mu (float): Shear modulus.
    
    Returns:
        E (float): Young's modulus.
        nu (float): Poisson's ratio.
    """
    E = 9 * kappa * mu / (3 * kappa + mu)  # Young's modulus (E)
    nu = (3 * kappa - 2 * mu) / (2 * (3 * kappa + mu))  # Poisson's ratio (nu)
    return E, nu


def E_nu_to_kappa_mu_2d_plane_strain(E, nu):
    """
    2D plane strain conversion.
    """
    mu = E / (2 * (1 + nu))
    kappa = E / (2 * (1 - nu))   # NOTE: different from 3D!
    return kappa, mu


def kappa_mu_to_E_nu_2d_plane_strain(kappa, mu):
    """
    2D plane strain conversion.
    """
    # E = 4 * mu * (kappa - mu) / (kappa + mu)
    # nu = (kappa - mu) / (kappa + mu)
    E = 4*mu*kappa/(kappa+mu)
    nu = (kappa-mu)/(kappa+mu)
    return E, nu


def extract_effective_moduli_3d(
        C_hom: np.ndarray,
):
    """
    Extracts effective moduli (kappa_eff, mu_eff, E_eff, nu_eff) from the homogenized stiffness matrix.
    """
    # Extract the relevant entries:
    C11, C22, C33 = C_hom[0, 0], C_hom[1, 1], C_hom[2, 2]
    C12, C13, C23 = C_hom[0, 1], C_hom[0, 2], C_hom[1, 2]
    C44, C55, C66 = C_hom[3, 3], C_hom[4, 4], C_hom[5, 5]

    # Effective lame moduli:
    lambda_eff = C12
    mu_eff = C44

    # Effective bulk modulus
    kappa_eff = lambda_eff + 2 / 3 * mu_eff

    # Convert to effective Young's modulus and Poisson's ratio:
    E_eff, nu_eff = kappa_mu_to_E_nu_3d(kappa_eff, mu_eff)

    return kappa_eff, mu_eff, E_eff, nu_eff


def extract_effective_moduli_2d_plane_strain(
        C_hom: np.ndarray,
):
    """
    Extract effective moduli from 2D homogenized stiffness (plane strain).
    Voigt order: [xx, yy, xy]
    """
    C11 = C_hom[0, 0]
    C22 = C_hom[1, 1]
    C12 = C_hom[0, 1]
    C33 = C_hom[2, 2]

    # Lame parameters
    lambda_eff = C12
    mu_eff = C33

    # Bulk modulus (2D plane strain)
    kappa_eff = lambda_eff + mu_eff

    # Convert
    E_eff, nu_eff = kappa_mu_to_E_nu_2d_plane_strain(kappa_eff, mu_eff)

    return kappa_eff, mu_eff, E_eff, nu_eff


def _get_isotropic_stiffness(E, nu, dim=3):
    if dim == 3:
        # 3D Stiffness Matrix (Voigt notation: 11, 22, 33, 23, 13, 12)
        C = np.zeros((6, 6))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        C[:3, :3] = lam
        np.fill_diagonal(C[:3, :3], lam + 2 * mu)
        np.fill_diagonal(C[3:, 3:], mu)
    else:
        # 2D Plane Strain (Voigt: 11, 22, 12)
        C = np.zeros((3, 3))
        factor = E / ((1 + nu) * (1 - 2 * nu))
        C[0,0] = C[1,1] = factor * (1 - nu)
        C[0,1] = C[1,0] = factor * nu
        C[2,2] = E / (2 * (1 + nu)) # mu
    return C


def _get_eshelby_tensor(nu, dim=3):
    S = np.zeros((6, 6) if dim == 3 else (3, 3))
    if dim == 3:
        # Eshelby for Sphere in 3D
        s1 = 7 - 5 * nu
        s2 = 8 - 10 * nu
        s3 = 15 * (1 - nu)
        S[0,0] = S[1,1] = S[2,2] = (7 - 5*nu) / s3
        S[0,1] = S[0,2] = S[1,0] = S[1,2] = S[2,0] = S[2,1] = (5*nu - 1) / s3
        S[3,3] = S[4,4] = S[5,5] = (4 - 5*nu) / s3
    else:
        # Eshelby for Circular Cylinder (Plane Strain)
        S[0,0] = S[1,1] = (3 - 4*nu) / (8 * (1 - nu))
        S[0,1] = S[1,0] = (4*nu - 1) / (8 * (1 - nu))
        S[2,2] = (1 - 2*nu) / (4 * (1 - nu))
    return S


def mori_tanaka(E_m, nu_m, E_f, nu_f, vol_f, dim=3):
    """
    Computes the effective stiffness tensor using the Mori-Tanaka homogenization scheme.

    This semi-analytical model accounts for the interaction between inclusions by 
    assuming each inclusion is embedded in the average stress field of the matrix. 
    It is more accurate than the dilute approximation for moderate volume fractions.

    Parameters
    ----------
    E_m, nu_m : float
        Young's modulus and Poisson's ratio of the matrix.
    E_f, nu_f : float
        Young's modulus and Poisson's ratio of the fiber/inclusion.
    vol_f : float
        Volume fraction of the inclusions (0.0 to 1.0).
    dim : int, default 3
        Spatial dimension (2 or 3).

    Returns
    -------
    np.ndarray
        The effective (homogenized) stiffness matrix C_eff in Voigt notation.
    """
    Cm = _get_isotropic_stiffness(E_m, nu_m, dim)
    Cf = _get_isotropic_stiffness(E_f, nu_f, dim)
    S = _get_eshelby_tensor(nu_m, dim)
    I = np.eye(S.shape[0])

    # Concentration Tensor A_pa (Partial)
    # A = [I + S : inv(Cm) : (Cf - Cm)]^-1
    invCm = np.linalg.inv(Cm)
    A_temp = I + S @ invCm @ (Cf - Cm)
    A_pa = np.linalg.inv(A_temp)

    # Effective Stiffness C_eff = Cm + f(Cf - Cm)*A_pa  [ (1-f)I + f*A_pa ]^-1
    f = vol_f
    term1 = f * (Cf - Cm) @ A_pa
    term2 = np.linalg.inv((1 - f) * I + f * A_pa)
    C_eff = Cm + term1 @ term2

    return C_eff


def hashin_shtrikman(E_m, nu_m, E_f, nu_f, vol_f, dim=3):
    """
    Computes the Hashin-Shtrikman upper and lower bounds for effective properties.

    These bounds represent the theoretical limits for any isotropic composite 
    without assuming a specific geometry (like circles or squares). For a 
    stiffer inclusion (E_f > E_m), the 'lower' bound corresponds to the 
    matrix-dominated case (Mori-Tanaka) and the 'upper' bound to the 
    inclusion-dominated case.

    Parameters
    ----------
    E_m, nu_m : float
        Properties of the matrix.
    E_f, nu_f : float
        Properties of the fiber/inclusion.
    vol_f : float
        Volume fraction of the inclusion.
    dim : int, default 3
        Spatial dimension (2 for Plane Strain, 3 for 3D).

    Returns
    -------
    dict
        A dictionary containing tuples of (Lower, Upper) bounds for Bulk modulus (kappa),
        Shear modulus (mu), Young's modulus (E), and Poisson's ratio (nu).
    """
    def get_kg(E, nu, d):
        if d == 3:
            K = E / (3 * (1 - 2 * nu))
            G = E / (2 * (1 + nu))
        else: # Plane Strain 2D
            K = E / (2 * (1 + nu) * (1 - 2 * nu))
            G = E / (2 * (1 + nu))
        return K, G

    K_m, G_m = get_kg(E_m, nu_m, dim)
    K_f, G_f = get_kg(E_f, nu_f, dim)

    f1 = 1 - vol_f
    f2 = vol_f

    if dim == 3:
        def hs_k(k1, g1, k2, g2, v2):
            return k1 + v2 / (1/(k2 - k1) + (3 * (1 - v2)) / (3 * k1 + 4 * g1))

        def hs_g(k1, g1, k2, g2, v2):
            term = (6 * (k1 + 2 * g1)) / (5 * g1 * (3 * k1 + 4 * g1))
            return g1 + v2 / (1/(g2 - g1) + 6 * (1 - v2) * term / 6)
    else:
        def hs_k(k1, g1, k2, g2, v2):
            return k1 + v2 / (1/(k2 - k1) + (1 - v2) / (k1 + g1))

        def hs_g(k1, g1, k2, g2, v2):
            return g1 + v2 / (1/(g2 - g1) + (k1 + 2*g1)*(1 - v2) / (2*g1*(k1 + g1)))

    K_low = hs_k(K_m, G_m, K_f, G_f, f2)
    K_high = hs_k(K_f, G_f, K_m, G_m, f1)

    G_low = hs_g(K_m, G_m, K_f, G_f, f2)
    G_high = hs_g(K_f, G_f, K_m, G_m, f1)

    if dim==2:
        E_low, nu_low = kappa_mu_to_E_nu_3d(K_low, G_low)
        E_high, nu_high = kappa_mu_to_E_nu_3d(K_high, G_high)
    elif dim==3:
        E_low, nu_low = kappa_mu_to_E_nu_2d_plane_strain(K_low, G_low)
        E_high, nu_high = kappa_mu_to_E_nu_2d_plane_strain(K_high, G_high)

    return {"kappa": (K_low, K_high), "mu": (G_low, G_high), "E": (E_low, E_high), "nu": (nu_low, nu_high)}


__all__ = [
    # moduli
    "E_nu_to_kappa_mu_3d",
    "E_nu_to_kappa_mu_2d_plane_strain",
    "kappa_mu_to_E_nu_3d",
    "kappa_mu_to_E_nu_2d_plane_strain",
    "extract_effective_moduli_3d",
    "extract_effective_moduli_2d_plane_strain",
    "mori_tanaka",
    "hashin_shtrikman",
]



