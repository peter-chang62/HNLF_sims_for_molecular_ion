# %% ----- package imports
import numpy as np
import matplotlib.pyplot as plt
import clipboard
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq
import pynlo
from scipy.constants import c
from scipy.interpolate import InterpolatedUnivariateSpline
import collections
from scipy.optimize import minimize

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


# units
nm = 1e-9
um = 1e-6
uW = 1e-6
mW = 1e-3
ps = 1e-12
fs = 1e-15
km = 1e3
THz = 1e12
W = 1.0

# %% ----- reconstructed pulse from Toptica -----------------------------------
path = "from_Toptica/"
recon_s = np.genfromtxt(path + "ReconstructedPulseSpectrum.txt")
recon_t = np.genfromtxt(path + "ReconstructedPulseTemporal.txt")

t_grid = recon_t[:, 2] * fs
wl_grid = recon_s[:, 2] * um
v_grid = c / wl_grid

p_v = recon_s[:, 0]
p_t = recon_t[:, 0] ** 2
phi_v = recon_s[:, 1]
phi_t = recon_t[:, 1]

# %% ----- pynlo pulse --------------------------------------------------------
f_r = 200e6
n = 256
v_min = c / 2e-6
v_max = c / 1e-6
v0 = c / (1550 * nm)
t_fwhm = 100e-15
min_time_window = t_grid[-1] - t_grid[0]
e_p = 1.7 / f_r
pulse = pynlo.light.Pulse.Sech(
    n=n,
    v_min=v_min,
    v_max=v_max,
    v0=v0,
    e_p=e_p,
    t_fwhm=t_fwhm,
    min_time_window=min_time_window,
)

# interpolate reconstructed envelope onto simulation grid
pulse.import_p_v(v_grid, p_v, phi_v)

# double check the interpolation -> pass!
# fig, ax = plt.subplots(1, 2)
# ax[0].plot(v_grid / THz, p_v / p_v.max())
# ax[0].plot(pulse.v_grid / THz, pulse.p_v / pulse.p_v.max())
# ax[0].set_xlabel("THz")
# ax[0].set_ylabel("arb.")
# ax[1].plot(t_grid / ps, p_t / p_t.max())
# ax[1].plot(pulse.t_grid / ps, pulse.p_t / pulse.p_t.max())
# ax[1].set_xlabel("time (ps)")
# ax[1].set_ylabel("arb.")
# fig.tight_layout()

# %% ----- pm1550 and hnlf ----------------------------------------------------
gamma_pm1550 = 1.2
pm1550 = pynlo.materials.SilicaFiber()
pm1550.load_fiber_from_dict(pynlo.materials.pm1550)
pm1550.gamma = gamma_pm1550 / (W * km)

hnlf = pynlo.materials.SilicaFiber()
hnlf.load_fiber_from_dict(pynlo.materials.hnlf_5p7)

# %% ----- maximize the psd at 1450 nm based on length of pm-1550 and hnlf
iteration = 0


def func(X, include_loss=True, return_output=False):
    # print iteration number and current fiber lengths
    global iteration
    print(iteration, X)
    iteration += 1

    # option to include fiber coupling loss
    p_in = pulse.copy()
    if include_loss:
        p_in.e_p *= 0.7  # 70% coupling efficiency into the fiber patch cable?

    # pm-1550 sim
    length_pm1550, length_hnlf = X * 1e-2
    output_pm1550 = propagate(pm1550, p_in, length_pm1550)
    sim_pm1550 = output_pm1550.sim

    # option to include splicing loss from PM-1550 -> HNLF
    p_in = sim_pm1550.pulse_out.copy()
    if include_loss:
        p_in.e_p *= 10 ** (-1 / 10)  # ~1 dB splicing loss between PM-1550 and HNLF

    # hnlf sim
    output_hnlf = propagate(hnlf, p_in, length_hnlf)
    sim_hnlf = output_hnlf.sim

    # get the psd (J/Hz) at 1450 nm
    p_v = sim_hnlf.pulse_out.p_v
    idx = abs(pulse.wl_grid - 1450e-9).argmin()
    psd = p_v[idx]

    if return_output:
        return output_pm1550, output_hnlf
    else:
        return -psd


res = minimize(fun=func, x0=np.array([5.5, 2]), method="Nelder-Mead")
output_pm1550, output_hnlf = func(res.x, return_output=True)

# %% --------------------------------------------------------------------------
fig, ax = plt.subplots(1, 1)
p_v = output_hnlf.sim.pulse_out.p_v
dv_dl = pulse.v_grid**2 / c
ax.plot(pulse.wl_grid / um, p_v * dv_dl * f_r / (mW / nm))
phi_v = np.unwrap(np.angle(output_hnlf.sim.pulse_out.a_v))
ax_2 = ax.twinx()
ax_2.plot(pulse.wl_grid / um, phi_v * 180 / np.pi, "C1")
ax.set_xlabel("wavelength ($\\mu m$)")
ax.set_ylabel("PSD (mW/nm)")
ax_2.set_ylabel("spectral phase (deg.)")
fig.tight_layout()
