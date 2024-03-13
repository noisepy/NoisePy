import matplotlib.pyplot as plt
import numpy as np


def plot_waveforms(ncmp, wav, fname, comp_arr):
    fig, ax = plt.subplots(1, ncmp, figsize=(16, 3), sharex=False)

    for n in range(ncmp):
        absy = max(wav[n][1], key=abs)
        ax[n].set_ylim(absy * -1, absy)
        ax[n].plot(wav[n][0], wav[n][1])
        ax[n].set_xlabel("time [s]")
        ax[n].set_title(fname + " " + comp_arr[n])
    fig.tight_layout()
    # print("save figure as Waveform_readin_%s.png"%(fname))
    plt.savefig("Waveform_readin_%s.png" % (fname), format="png", dpi=100)
    plt.close(fig)


def plot_filtered_waveforms(freq, tt, wav, fname, ccomp):
    nfreq = len(freq) - 1
    fig, ax = plt.subplots(1, nfreq, figsize=(16, 3), sharex=False)

    for fb in range(nfreq):
        fmin = freq[fb]
        fmax = freq[fb + 1]
        absy = max(wav[fb], key=abs)
        ax[fb].set_ylim(absy * -1, absy)
        ax[fb].plot(tt, wav[fb], "k-", linewidth=0.2)
        ax[fb].set_xlabel("Time [s]")
        ax[fb].set_ylabel("Amplitude")
        ax[fb].set_title("%s   %s   @%4.2f-%4.2f Hz" % (fname, ccomp, fmin, fmax))
    fig.tight_layout()
    plt.savefig("Waveform_filtered_%s_%s_F%s-%s.png" % (fname, ccomp, fmin, fmax), format="png", dpi=100)
    plt.close(fig)


def plot_envelope(comp_arr, freq, msv, msv_mean, fname, vdist):
    nfreq = len(freq) - 1
    ncmp = len(comp_arr)

    fig, ax = plt.subplots(ncmp + 1, nfreq, figsize=(16, 10), sharex=False)
    for n in range(len(comp_arr)):
        for fb in range(nfreq):
            fmin = freq[fb]
            fmax = freq[fb + 1]
            ax[n, fb].plot(msv[n][0][:], msv[n][fb + 1], "k-", linewidth=0.5)
            ax[n, fb].set_title("%s   %.2fkm  %s   @%4.2f-%4.2f Hz" % (fname, vdist, comp_arr[n], fmin, fmax))
            ax[n, fb].set_xlabel("Time [s]")
            ax[n, fb].set_ylabel("Amplitude")

    for fb in range(nfreq):
        fmin = freq[fb]
        fmax = freq[fb + 1]
        ax[-1, fb].plot(msv_mean[0], msv_mean[fb + 1], "b-", linewidth=1)
        ax[-1, fb].set_title(" Mean Squared Value %.2fkm  @%4.2f-%4.2f Hz" % (vdist, fmin, fmax))
        ax[-1, fb].set_xlabel("Time [s]")
        ax[-1, fb].set_ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig("Waveform_envelope_%s_F%s-%s.png" % (fname, fmin, fmax), format="png", dpi=100)
    plt.close(fig)


def plot_fmsv_waveforms(freq, wav, fname, noise_level, twin):
    nfreq = len(freq) - 1
    fig, ax = plt.subplots(1, nfreq, figsize=(16, 3), sharex=False)

    for fb in range(nfreq):
        fmin = freq[fb]
        fmax = freq[fb + 1]
        absy = 1  # max(wav[fb], key=abs)
        ax[fb].plot(
            [wav[0][0], wav[0][-1]], [noise_level[fb], noise_level[fb]], c="blue", marker=".", ls="--", linewidth=2
        )

        ax[fb].plot([twin[fb][0], twin[fb][0]], [-0.1, absy], c="orange", marker=".", ls="--", linewidth=2)
        ax[fb].plot([twin[fb][1], twin[fb][1]], [-0.1, absy], c="orange", marker=".", ls="--", linewidth=2)
        ax[fb].set_yscale("log", base=10)
        ax[fb].plot(wav[0], wav[fb + 1], "k-", linewidth=0.5)
        ax[fb].set_xlabel("Time [s]")
        ax[fb].set_ylabel("Amplitude in log-scale")
        ax[fb].set_title("%s   @%4.2f-%4.2f Hz" % (fname, fmin, fmax))
    fig.tight_layout()
    plt.savefig("Waveform_fmsv_%s.png" % (fname), format="png", dpi=100)
    plt.close(fig)


def plot_fitting_curves(mean_free, intrinsic_b, tt, Eobs, Esyn, fname, dist, twin, fmin, fmax):
    numb = len(intrinsic_b)
    plt.figure(figsize=(8, 2))
    for nb in range(numb):
        plt.yscale("log", base=10)
        # plt.xlim(0,120)
        pymin = np.min(Eobs[nb][:-2] / 2)
        pymax = np.max(Eobs[nb][:-2] * 2)
        plt.ylim(pymin, pymax)
        plt.plot(tt, Eobs[nb], "k-", linewidth=0.5)
        plt.plot(tt, Esyn[nb], "b-", linewidth=1)
        plt.plot([twin[0], twin[0], twin[-1], twin[-1], twin[0]], [pymin, pymax, pymax, pymin, pymin], "r", linewidth=2)

    plt.title(
        "%s  %.2fkm   @%4.2f-%4.2f Hz, mean_free: %.2f  b: %.2f~%.2f"
        % (fname, dist, fmin, fmax, mean_free, intrinsic_b[0], intrinsic_b[-1])
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Energy density Amplitude")
    plt.tight_layout()
    plt.savefig("Fitting_fmsv_%s_F%s-%s_MFP%.2f.png" % (fname, fmin, fmax, mean_free), format="png", dpi=100)
    plt.close()


def plot_fitting_result(mean_free, intrinsic_b, tt, Eobs, Esyn, fname, dist, twin, fmin, fmax):
    plt.figure(figsize=(6, 2))
    plt.yscale("log", base=10)

    pymax = np.max(Eobs[:-2] * 5)
    pymin = 10 ** (-6)
    plt.ylim(pymin, pymax)
    plt.plot(tt, Eobs, "k-", linewidth=1)
    plt.plot(tt, Esyn, "b--", linewidth=1)
    plt.plot([twin[0], twin[0], twin[-1], twin[-1], twin[0]], [pymin, pymax, pymax, pymin, pymin], "r", linewidth=2)

    plt.title("%s  %.2fkm   @%4.2f-%4.2f Hz, intrinsic b: %.2f" % (fname, dist, fmin, fmax, intrinsic_b))
    plt.xlabel("Time [s]")
    plt.ylabel("Energy density Amp")
    plt.tight_layout()
    plt.savefig("Final_fmsv_%s_F%s-%s.png" % (fname, fmin, fmax), format="png", dpi=100)
