# %% ----- package imports
import numpy as np
import matplotlib.pyplot as plt
import clipboard
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq
import pynlo
from scipy.constants import c
import collections
from scipy.optimize import minimize
from tqdm import tqdm

forward_transform = lambda x, d=1: fftshift(fft(ifftshift(x))) * d
inverse_transform = lambda x, d=1: fftshift(ifft(ifftshift(x))) / d

output = collections.namedtuple("output", ["model", "sim"])


def propagate(fiber, pulse, length, n_records=100):
    """
    propagates a given pulse through fiber of given length

    Args:
        fiber (instance of SilicaFiber): Fiber
        pulse (instance of Pulse): Pulse
        length (float): fiber elngth

    Returns:
        output: model, sim
    """
    model = fiber.generate_model(pulse, t_shock="auto", raman_on=True)
    dz = model.estimate_step_size()
    sim = model.simulate(length, dz=dz, n_records=n_records)
    return output(model=model, sim=sim)


km = 1e3
cm = 1e-2
um = 1e-6
nm = 1e-9
ps = 1e-12
W = 1.0

# %% ----- pynlo pulse --------------------------------------------------------
f_r = 200e6  # repetition rate
v_min = c / (2 * um)  # minimum frequency of grid
v_max = c / (1 * um)  # maximum frequency of grid
v0 = c / (1550 * nm)  # center frequency (carrier)
min_time_window = 10e-12  # 10 pstime window
input_power = 1.2  # Watts

pulse = pynlo.light.Pulse.Sech(
    n=256,  # number of points for simulation grid, will be automatically updated
    v_min=v_min,
    v_max=v_max,
    v0=v0,
    e_p=input_power / f_r,
    t_fwhm=100e-15,  # fwhm for initial Sech pulse, will be overriden by reconstruction
    min_time_window=min_time_window,
)

# %% ----- reconstructed pulse from Toptica -----------------------------------
# data is structured as time/wavelength, intensity, phase
# the spectrum reconstruction is linearly spaced in frequency, not wavelength,
# although the axis is given in wavelength

recon_s = np.genfromtxt(
    "from_Toptica/correct_frequency_reconstruction.txt", skip_header=1
)
recon_t = np.genfromtxt(
    "from_Toptica/correct_temporal_reconstruction.txt", skip_header=1
)

wl_grid = recon_s[:, 0] * nm
v_grid = c / wl_grid
p_wl = recon_s[:, 1]
p_v = p_wl / (v_grid**2 / c)
phi_v = recon_s[:, 2]

t_grid = recon_t[:, 0] * 1e-12
p_t = recon_t[:, 1]
phi_t = recon_t[:, 2]

pulse.import_p_v(v_grid, p_v, -phi_v)  # interoplate reconstruction onto pulse grid

# %% ----- pm1550 and hnlf ----------------------------------------------------
gamma_pm1550 = 1.2
pm1550 = pynlo.materials.SilicaFiber()
pm1550.load_fiber_from_dict(pynlo.materials.pm1550)  # sets dispersion and gamma
pm1550.gamma = gamma_pm1550 / (W * km)

adhnlf = pynlo.materials.SilicaFiber()
ndhnlf = pynlo.materials.SilicaFiber()
hnlf_dict = {
    "D slow axis": -2.2 * ps / (nm * km),
    "D slope slow axis": 0.026 * ps / (nm**2 * km),
    "D fast axis": 1.0 * ps / (nm * km),
    "D slope fast axis": 0.024 * ps / (nm**2 * km),
    "nonlinear coefficient": 10.5 / (W * km),
    "center wavelength": 1550 * nm,
}
adhnlf.load_fiber_from_dict(pynlo.materials.hnlf_5p7)  # sets dispersion and gamma
ndhnlf.load_fiber_from_dict(hnlf_dict)  # sets dispersion and gamma

# %% ----- reference osa data -------------------------------------------------
s = np.genfromtxt("data/W0008.CSV", delimiter=",", skip_header=29)
v_grid = c / (s[:, 0] * nm)
s[:, 1] = 10 ** (s[:, 1] / 10)
pulse_ref = pulse.copy()
pulse_ref.import_p_v(v_grid, s[:, 1] / (v_grid**2 / c))
(idx_osa,) = np.logical_and(
    v_grid.min() < pulse.v_grid, pulse.v_grid < v_grid.max()
).nonzero()


# %% ----- simulation function ------------------------------------------------
# simulate propagation through pm-1550 -> hnlf with 1 dB splicing loss at given
# input power, length of pm-1550, and length of hnlf.
def simulate(power, length_pm1550, length_hnlf, plot=True):
    length_pm1550 *= cm
    length_hnlf *= cm
    p_in = pulse.copy()
    p_in.e_p = power / f_r

    out_pm1550 = propagate(pm1550, p_in, length_pm1550)
    p_pm1550 = out_pm1550.sim.pulse_out
    p_pm1550.e_p *= 10 ** (-1 / 10)  # 1 dB splicing loss

    out_adhnlf = propagate(adhnlf, p_pm1550, length_hnlf)
    p_adhnlf = out_adhnlf.sim.pulse_out

    # compare output spectrum to reference spectrum.
    # account for an overall scale difference by minimizing the rms error with
    # overall scaling.
    p_ref = pulse_ref.copy()
    p_ref.e_p = p_adhnlf.e_p
    x1 = p_ref.p_v[idx_osa].copy()
    x2 = p_adhnlf.p_v[idx_osa].copy()
    f = lambda s: np.sqrt(np.mean(abs(x1 * s - x2) ** 2))
    res = minimize(f, 1, method="Nelder-Mead")
    if plot:
        plt.gca().clear()
        plt.plot(x1, "k", linewidth=2)
        plt.plot(x2)
        plt.pause(0.01)

    return out_adhnlf, res


# %% ----- try matching the osa spectrum by varying input power ---------------
def func_power(length_pm1550, length_hnlf):
    def f(power):
        print(power)
        return simulate(
            power,
            length_pm1550,
            length_hnlf,
            plot=True,
        )[1].fun

    return f


res = minimize(func_power(5.5, 2.0), np.array([1.2]), method="Nelder-Mead")
out, res_scale = simulate(res.x, 5.5, 2.0, True)
scale = res_scale.x

# %% ----- plotting -----------------------------------------------------------
p_ref = pulse_ref.copy()
p_ref.e_p = out.sim.pulse_out.e_p
fig, ax = plt.subplots(1, 1)
ax.plot(
    pulse.wl_grid[idx_osa] * 1e9,
    p_ref.p_v[idx_osa] * scale * pulse.v_grid[idx_osa] ** 2 / c,
    label="experimental",
)
ax.plot(
    pulse.wl_grid[idx_osa] * 1e9,
    out.sim.pulse_out.p_v[idx_osa] * pulse.v_grid[idx_osa] ** 2 / c,
    label="simulated",
)
ax.grid(alpha=0.25)
ax_2 = ax.twinx()
ax_2.plot(
    pulse.wl_grid[idx_osa] * 1e9,
    np.unwrap(out.sim.pulse_out.phi_v[idx_osa]) * 180 / np.pi,
    "C3",
    label="phase",
)
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("psd (arb.)")
ax_2.set_ylabel("phase (deg.)")
handles_1, labels_1 = ax.get_legend_handles_labels()
handles_2, labels_2 = ax_2.get_legend_handles_labels()
handles = handles_1 + handles_2
labels = labels_1 + labels_2
ax.legend(loc="best", handles=handles, labels=labels)
fig.tight_layout()
