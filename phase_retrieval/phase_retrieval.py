# %% ----- imports
import numpy as np
import matplotlib.pyplot as plt
import clipboard
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq
from scipy.constants import c
from collections import namedtuple
from scipy.interpolate import InterpolatedUnivariateSpline, RegularGridInterpolator
import scipy
from tqdm import tqdm
import blit
from BBO import BBOSHG as BBO

# %% --------------------------------------------------------------------------
file = "edfa_out.npz"
file_bckgnd = "edfa_out_bckgnd.npz"

level_of_marginal_t = 1 / np.exp(1)
factor_bandwidth_v = 1.25

# %% --------------------------------------------------------------------------
output = namedtuple("frog_data", ["t_grid", "wl_grid", "s"])


def forward_transform(x, dx=1.0, axis=0):
    return fftshift(fft(ifftshift(x, axes=axis), axis=axis), axes=axis) * dx


def inverse_transform(x, dx=1.0, axis=0):
    return fftshift(ifft(ifftshift(x, axes=axis), axis=axis), axes=axis) / dx


def shg_frog(a_t, dt=1.0):
    o = a_t * np.c_[a_t]
    o_rs = np.zeros_like(o)
    for r in range(o.shape[0]):
        o_rs[r] = np.roll(o[r], -r)
    s_t = fftshift(o_rs, axes=1)
    s_v = forward_transform(s_t, dx=dt, axis=0)
    return s_v


def load_file(file):
    data = np.load(file)
    return output(data["t_grid"] * 1e-15, data["wl_grid"] * 1e-9, data["spectrogram"])


def shift(array, dx):
    ft = forward_transform(array)
    freq = fftshift(fftfreq(array.size))
    omega = 2 * np.pi * freq
    ft *= np.exp(1j * omega * dx)
    return inverse_transform(ft)


def soft_threshold(x, gamma):
    return np.where(x < gamma, 0, x - gamma * np.sign(x))


# %% ----- load data and background -------------------------------------------
data = load_file(file)
bckgnd = load_file(file_bckgnd)

# background subtraction
s = data.s - bckgnd.s * 1.01
s = np.where(s < 0, 0, s)

# 1 um pump leak through
# s[:, 1500:] = 0

# center frog trace in time
marginal_t = np.sum(s, axis=1)
idx = marginal_t.argmax()
n = min([idx, data.t_grid.size - idx]) * 2
n = n if n % 2 == 0 else n - 1
s = s[idx - n // 2 : idx + n // 2]
t_grid = data.t_grid - data.t_grid[idx]
t_grid = t_grid[idx - n // 2 : idx + n // 2]

# create interpolation object
v_grid = c / data.wl_grid
s_v_interp = RegularGridInterpolator(
    (t_grid, v_grid),
    s * c / v_grid**2,
    bounds_error=False,
    fill_value=0.0,
)

# %% ------ define retrieval grid and interpolate frog onto grid --------------
marginal_v = np.sum(s, axis=0)
marginal_v /= marginal_v.max()

# find 20dB width of frequency marginal, the frequency bandwidth is set to 20dB
# width * factor_bandwidth_v
marginal_v_interp = InterpolatedUnivariateSpline(v_grid[::-1], marginal_v[::-1] - 0.01)
roots_v = marginal_v_interp.roots()
roots_v = roots_v[[0, -1]]
bandwidth_v = np.diff(roots_v) * factor_bandwidth_v
bandwidth_t = t_grid.max() - t_grid.min()
v0 = (roots_v[1] - roots_v[0]) / 2 + roots_v[0]

# find number of points needed to fill time and frequency grid
# interpolate the experimental frog trace onto this grid
n_points = int(np.ceil(bandwidth_t * bandwidth_v))
n_points = n_points if n_points % 2 == 0 else n_points + 1
t_grid_new = np.linspace(-bandwidth_t / 2, bandwidth_t / 2, n_points)
dt = t_grid_new[1] - t_grid_new[0]
v_grid_new = fftshift(fftfreq(n_points, dt)) + v0
T_grid_new, V_grid_new = np.meshgrid(t_grid_new, v_grid_new)
s_v_new = s_v_interp((T_grid_new, V_grid_new)).T
s_v_new /= s_v_new.max()

# %% ----- divide by phasematching curve --------------------------------------
# honestly it's basically flat
bbo = BBO()
R = bbo.R(
    wl_um=c / v_grid_new * 1e6,
    length_um=50,
    theta_pm_rad=bbo.phase_match_angle_rad(1.55),
    alpha_rad=np.arctan(
        0.25 / 6,
    ),
)
s_v_new /= R
s_v_new /= s_v_new.max()

# %% ----- list of time windows moving out by, e.g. fwhm, from center ---------
marginal_t = np.sum(s_v_new, axis=1)
marginal_t /= marginal_t.max()
marginal_t_interp = InterpolatedUnivariateSpline(
    t_grid_new, marginal_t - level_of_marginal_t
)
roots_t = marginal_t_interp.roots()

idx_0 = abs(t_grid_new - roots_t[0]).argmin()
idx_1 = abs(t_grid_new - roots_t[-1]).argmin()
idx_width = idx_1 - idx_0
idx_width = idx_width if idx_width % 2 == 0 else idx_width + 1

center = n_points // 2
N_windows = int(np.ceil(n_points / idx_width))
idx_subset_list = []
idx_full = np.arange(n_points)
for i in range(1, N_windows + 1):
    start = center - i * idx_width // 2
    start = 0 if start < 0 else start

    end = center + i * idx_width // 2
    end = n_points if end > n_points else end

    subset = idx_full[start:end]
    idx_subset_list.append(subset)

# %% ----- set initial guess --------------------------------------------------
E_i = scipy.signal.gaussian(n_points, idx_width).astype(complex)
E_i_best = np.zeros_like(E_i)

fig, ax = plt.subplots(1, 1)
[ax.plot(i, np.zeros_like(i) + n, "C0") for n, i in enumerate(idx_subset_list)]
ax.plot(marginal_t * len(idx_subset_list), "C1")
ax.plot(E_i * len(idx_subset_list), "C2", linestyle="--")

# %% --------------------------------------------------------------------------
# np.random.seed(1)

N_iter = 300
plot_initialized = False
for n, idx_subset in enumerate(tqdm(idx_subset_list)):
    # select the best result from the previous cropped time window?
    # Sometimes works, but tends to hurt more than it helps, I think because it
    # keeps the algorithm from searching the full parameter space.
    # if n > 0:
    #     E_i[:] = E_i_best[:]
    for i in tqdm(range(N_iter)):
        idx_subset_scrambled = np.random.permutation(idx_subset)
        alpha = np.random.uniform(low=0.1, high=0.5)

        for k in idx_subset_scrambled:
            delay = k - center

            E_i_k = shift(E_i, delay)
            Psi_i_k = E_i * E_i_k
            Phi_i_k = forward_transform(Psi_i_k, dx=dt)
            phase = np.unwrap(np.angle(Phi_i_k))
            Phi_i_k_new = np.sqrt(s_v_new[k]) * np.exp(1j * phase)

            # snr soft thresholding...

            Psi_i_k_new = inverse_transform(Phi_i_k_new, dx=dt)

            factor = Psi_i_k_new - Psi_i_k
            obj_update = np.conj(E_i_k) / (abs(E_i_k) ** 2).max() * factor
            probe_update = np.conj(E_i) / (abs(E_i) ** 2).max() * factor
            probe_update = shift(probe_update, -delay)

            E_i += alpha * (obj_update + probe_update)
            E_i = np.roll(E_i, center - (abs(E_i) ** 2).argmax())

        # error calculation
        s_v_k = (abs(shg_frog(E_i, dt)) ** 2).T

        # calculate error based on the subset of the spectrogram being
        # retrieved
        num = np.sqrt(np.mean((s_v_k[idx_subset] - s_v_new[idx_subset]) ** 2))
        denom = np.sqrt(np.mean(s_v_new[idx_subset] ** 2))
        error = num / denom

        if i == 0:
            # initialize
            error_best = error
            E_i_best[:] = E_i[:]

        else:
            if error < error_best:
                E_i_best[:] = E_i[:]
                error_best = error
                # print(error_best)
                # print("updated")

                s_v_recon = (abs(shg_frog(E_i_best, dt)) ** 2).T
                p_t = abs(E_i_best) ** 2
                a_v = forward_transform(E_i_best, dt)
                p_v = abs(a_v) ** 2
                phi = np.unwrap(np.angle(a_v))
                phi -= phi[n_points // 2]
                phi *= 180 / np.pi
                if not plot_initialized:
                    fig, ax = plt.subplots(2, 2)
                    img_1 = ax[0, 0].imshow(
                        s_v_recon.T / s_v_recon.max(),
                        extent=(
                            t_grid_new[0] * 1e15,
                            t_grid_new[-1] * 1e15,
                            v_grid_new[0] * 1e-12,
                            v_grid_new[-1] * 1e-12,
                        ),
                        aspect="auto",
                        origin="lower",
                    )
                    img_2 = ax[0, 1].imshow(
                        s_v_new.T / s_v_new.max(),
                        extent=(
                            t_grid_new[0] * 1e15,
                            t_grid_new[-1] * 1e15,
                            v_grid_new[0] * 1e-12,
                            v_grid_new[-1] * 1e-12,
                        ),
                        aspect="auto",
                        origin="lower",
                    )

                    (l_1,) = ax[1, 0].plot(t_grid_new * 1e15, p_t / p_t.max())
                    (l_2,) = ax[1, 1].plot(v_grid_new * 1e-12, p_v / p_v.max())
                    ax_2 = ax[1, 1].twinx()
                    (l_3,) = ax_2.plot(v_grid_new * 1e-12, phi, "C1")

                    ax[0, 0].set_xlabel("time (fs)")
                    ax[0, 0].set_ylabel("frequency (THz)")
                    ax[0, 1].set_xlabel("time (fs)")
                    ax[0, 1].set_ylabel("frequency (THz)")
                    ax[1, 0].set_xlabel("time (fs)")
                    ax[1, 1].set_xlabel("freuqency (THz)")
                    ax_2.set_ylim(-360 * 2, 360 * 2)

                    fig.tight_layout()

                    bm = blit.BlitManager(fig.canvas, [img_1, img_2, l_1, l_2, l_3])
                    bm.update()

                    plot_initialized = True

                if plot_initialized:
                    img_1.set_data(s_v_recon.T / s_v_recon.max())
                    l_1.set_ydata(p_t / p_t.max())
                    l_2.set_ydata(p_v / p_v.max())
                    l_3.set_ydata(phi)
                    bm.update()

            else:
                # go back to the best one? This can hurt in that it doesn't
                # allow the algorithm to search?
                # E_i[:] = E_i_best[:]
                pass
print("best error:", error_best)

# %% --------------------------------------------------------------------------
# np.savez(
#     "retrieval_edfda_out_try2.npz",
#     t_grid=t_grid_new,
#     v_grid=v_grid_new,
#     a_t=E_i_best,
#     a_v=a_v,
# )
# plt.savefig("retrieval_edfda_out_try2.png", dpi=300)
