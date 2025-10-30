import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import base64
from gwpy.timeseries import TimeSeries
import matplotlib as mpl
mpl.use("agg")

st.set_page_config(page_title="GW Quickview")

st.title("Gravitational Wave Quickview (L1 & V1)")
st.markdown("Visualize time series, FFT, PSD, and spectrogram around a GPS trigger.")

@st.cache_data(max_entries=5)
def load_gw(t0, detector, fs=4096, pad=30):
    return TimeSeries.fetch_open_data(detector, t0 - pad, t0 + pad,
                                      sample_rate=fs, cache=False)

detectors = ["L1", "V1"]

# Sidebar controls
st.sidebar.header("Input Parameters")
t0 = float(st.sidebar.text_input("GPS Time", "1126259462.4"))
fs = st.sidebar.selectbox("Sample rate (Hz)", [4096, 16384], index=0)
whiten = st.sidebar.checkbox("Whiten", value=True)
freqrange = st.sidebar.slider("Band-pass range (Hz)", 10, 2000, (30, 400))
vmax = st.sidebar.slider("Spectrogram colorbar max", 10, 500, 25)
qcenter = st.sidebar.slider("Q-value", 5, 120, 5)
qrange = (int(qcenter * 0.8), int(qcenter * 1.2))

t_window = 0.5       # ± window for short data
welch_dur = 32.0     # seconds before trigger for PSD
pad = 33.0            # a bit extra margin for fetch

st.sidebar.markdown("---")
st.sidebar.write("Plots use ±0.5 s around trigger; PSD uses 32 s before trigger.")

strain_load_state = st.text("Loading data...")
strains = {}

for det in detectors:
    try:
        strains[det] = load_gw(t0, det, fs, pad)
    except Exception as e:
        st.warning(f"{det} data unavailable: {e}")

strain_load_state.text("Data loaded.")

# === Per detector plots ===
for det, strain_data in strains.items():
    st.header(f"Detector {det}")
    strain = deepcopy(strain_data)

    # crop ±0.5 s
    cropstart, cropend = t0 - t_window, t0 + t_window
    short = strain.crop(cropstart, cropend)

    # --- Raw time series ---
    st.subheader("Raw Strain ±0.5 s")
    fig_ts, ax = plt.subplots()
    ax.plot(short.times.value, short.value, lw=0.7)
    ax.set_xlabel("GPS time [s]")
    ax.set_ylabel("Strain")
    ax.grid(True, ls=":")
    st.pyplot(fig_ts, clear_figure=True)

    # --- FFT amplitude spectrum ---
    st.subheader("Amplitude Spectrum")
    y = short.value
    n = len(y)
    freq = np.fft.rfftfreq(n, 1.0 / fs)
    amp = np.abs(np.fft.rfft(y))
    fig_fft, ax = plt.subplots()
    ax.loglog(freq, amp)
    ax.set_xlim(10, fs / 2)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("|FFT|")
    ax.grid(True, which="both", ls=":")
    st.pyplot(fig_fft, clear_figure=True)

    # --- Whitened + Bandpass ---
    if whiten:
        bp_data = strain.whiten().bandpass(freqrange[0], freqrange[1])
    else:
        bp_data = strain.bandpass(freqrange[0], freqrange[1])
    bp_cropped = bp_data.crop(cropstart, cropend)
    st.subheader("Whitened & Band-passed Strain ±0.5 s")
    fig_bp = bp_cropped.plot()
    st.pyplot(fig_bp, clear_figure=True)

    # --- Welch PSD 32 s before trigger ---
    st.subheader("Median Welch PSD (32 s before trigger)")
    ref_start = t0 - welch_dur
    ref_end = t0
    try:
        ref = TimeSeries.fetch_open_data(det, ref_start, ref_end,
                                         sample_rate=fs, cache=False)
    except Exception:
        ref = strain_data

    psd_ref = ref.asd(fftlength=4, method="median")
    fig_psd, ax = plt.subplots()
    ax.loglog(psd_ref.frequencies.value, psd_ref.value, lw=1.2, label="Median Welch PSD")
    ax.set_xlim(10, fs / 2)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("ASD [strain/√Hz]")
    ax.legend()
    ax.grid(True, which="both", ls=":")
    st.pyplot(fig_psd, clear_figure=True)

    # --- Spectrogram ±0.5 s ---
    st.subheader("Spectrogram ±0.5 s")
    q = strain.q_transform(outseg=(t0 - t_window, t0 + t_window), qrange=qrange)
    fig_q = q.plot()
    ax = fig_q.gca()
    fig_q.colorbar(label="Normalized energy", vmax=vmax, vmin=0)
    ax.grid(False)
    ax.set_yscale("log")
    ax.set_ylim(bottom=15)
    st.pyplot(fig_q, clear_figure=True)

st.markdown("""
---
Data from the [Gravitational-Wave Open Science Center](https://gwosc.org).
""")
