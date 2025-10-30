import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from scipy.signal import tukey, welch
import matplotlib as mpl
mpl.use("agg")

st.set_page_config(page_title="GW Quickview", layout="wide")

st.title("Gravitational Wave Quickview (L1 & V1)")
st.markdown("Plots ±0.5 s around the trigger and 32 s of baseline for PSD estimation.")

# ---------------------------------------------------------------------
@st.cache_data(max_entries=5)
def load_gw(t0, det, fs=4096, pad=33):
    return TimeSeries.fetch_open_data(det, t0 - pad, t0 + pad,
                                      sample_rate=fs, cache=False)

# ---------------------------------------------------------------------
# Sidebar controls
t0 = float(st.sidebar.text_input("GPS time", "1126259462.4"))
fs = st.sidebar.selectbox("Sample rate (Hz)", [4096, 16384], index=0)
whiten = st.sidebar.checkbox("Whiten", True)
fband = st.sidebar.slider("Band-pass (Hz)", 10, 2000, (30, 400))
vmax = st.sidebar.slider("Spectrogram vmax", 10, 500, 25)
qcenter = st.sidebar.slider("Q-value", 5, 120, 5)
qrange = (int(qcenter*0.8), int(qcenter*1.2))

t_window = 0.5
psd_baseline = 32.0
detectors = ["L1", "V1"]

# ---------------------------------------------------------------------
# Load data
st.text("Loading strain data ...")
strains = {}
for det in detectors:
    try:
        strains[det] = load_gw(t0, det, fs)
    except Exception as e:
        st.warning(f"{det} unavailable: {e}")
st.success("Loaded.")

# ---------------------------------------------------------------------
# Helper to get ±0.5 s crop
def crop(ts):
    return ts.crop(t0 - t_window, t0 + t_window)

# ---------------------------------------------------------------------
# --- 1. Raw time series (overlaid)
st.header("1. Time series (±0.5 s)")
fig, ax = plt.subplots()
for det, ts in strains.items():
    seg = crop(ts)
    ax.plot(seg.times.value - t0, seg.value, lw=0.8, label=det)
ax.set_xlabel("Time relative to trigger [s]")
ax.set_ylabel("Strain")
ax.legend()
ax.grid(True, ls=":")
st.pyplot(fig, clear_figure=True)

# ---------------------------------------------------------------------
# --- 2. Whitened + Band-passed (overlaid)
st.header("2. Whitened & Band-passed time series (±0.5 s)")
fig, ax = plt.subplots()
for det, ts in strains.items():
    x = ts.whiten().bandpass(fband[0], fband[1]) if whiten else ts.bandpass(fband[0], fband[1])
    seg = crop(x)
    ax.plot(seg.times.value - t0, seg.value, lw=0.8, label=det)
ax.set_xlabel("Time relative to trigger [s]")
ax.set_ylabel("Strain (filtered)")
ax.legend()
ax.grid(True, ls=":")
st.pyplot(fig, clear_figure=True)

# ---------------------------------------------------------------------
# --- 3. PSD + Periodogram (overlaid)
st.header("3. Power Spectral Density (32 s baseline) + Periodogram (±0.5 s)")
fig, ax = plt.subplots()

for det, ts in strains.items():
    # Welch median PSD from 32 s before event
    ref = TimeSeries.fetch_open_data(det, t0 - psd_baseline, t0,
                                     sample_rate=fs, cache=False)
    psd_ref = ref.psd(fftlength=4, method="median")
    f_ref = psd_ref.frequencies.value
    asd_ref = np.sqrt(psd_ref.value)

    # Periodogram from ±0.5 s Tukey-windowed data
    short = crop(ts)
    y = short.value * tukey(len(short.value), 0.25)
    f_per, Pxx = welch(y, fs=fs, window='hann', nperseg=len(y), noverlap=0,
                       scaling='density')
    asd_per = np.sqrt(Pxx)

    ax.loglog(f_ref, asd_ref, lw=1.0, label=f"{det} Welch (32 s baseline)")
    ax.loglog(f_per, asd_per, lw=1.2, ls="--", label=f"{det} Periodogram (±0.5 s)")

ax.set_xlim(10, fs/2)
ax.set_ylim(1e-24, 1e-20)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("ASD [strain / √Hz]")
ax.legend(fontsize=8)
ax.grid(True, which="both", ls=":")
st.pyplot(fig, clear_figure=True)

# ---------------------------------------------------------------------
# --- 4. Spectrograms side by side
st.header("4. Spectrograms (±0.5 s)")

cols = st.columns(len(detectors))
for i, det in enumerate(detectors):
    if det not in strains:
        continue
    q = strains[det].q_transform(outseg=(t0 - t_window, t0 + t_window), qrange=qrange)
    fig_q = q.plot(figsize=(5, 3))
    ax = fig_q.gca()
    fig_q.colorbar(label="Normalized energy", vmax=vmax, vmin=0)
    ax.set_title(det)
    ax.set_xlabel("Time [s]")
    ax.set_yscale("log")
    ax.set_ylim(bottom=15)
    ax.grid(False)
    cols[i].pyplot(fig_q, clear_figure=True)

st.markdown("---")
st.markdown("Data from the [Gravitational-Wave Open Science Center](https://gwosc.org).")
