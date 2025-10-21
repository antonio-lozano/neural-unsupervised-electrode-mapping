import numpy as np
from typing import Tuple


def cartesian_to_polar(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r = np.sqrt(x * x + y * y)
    phi = np.arctan2(y, x)
    return r, phi


def complex_to_cartesian(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.real(z), np.imag(z)


def polar_to_complex(r: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return r * np.exp(1j * phi)


def cartesian_to_complex(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x + 1j * y


def complex_to_polar(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.abs(z), np.angle(z)


def f(z, a, b, alpha, k):
    def wedge(r, phi):
        return polar_to_complex(r, alpha * phi)

    def dipole(z):
        return k * np.log(b * (z + a) / (a * (z + b)))

    return dipole(wedge(*complex_to_polar(z)))


def visual_to_cortex_mapping(x, y, a, b, alpha, k):
    z = cartesian_to_complex(x, y)
    cortex_z = f(z, a, b, alpha, k)
    cortex_x, cortex_y = complex_to_cartesian(cortex_z)
    cortex_y = -1 * cortex_y
    return cortex_x, cortex_y


def f_inv(w, a, b, alpha, k):
    def wedge_inverse(z):
        r, phi = complex_to_polar(z)
        return polar_to_complex(r, phi / alpha)

    def dipole_inverse(w):
        e = np.exp(w / k)
        return a * b * (e - 1) / (b - a * e)

    return wedge_inverse(dipole_inverse(w))


def cortex_to_visual_mapping(x, y, a, b, alpha, k):
    w = cartesian_to_complex(x, -1 * y)
    visual_z = f_inv(w, a, b, alpha, k)
    visual_x, visual_y = complex_to_cartesian(visual_z)
    return visual_x, visual_y

