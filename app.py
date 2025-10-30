import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import base64

from gwpy.timeseries import TimeSeries
import matplotlib as mpl
mpl.use("agg")
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

st.set_page_config(page_title="GW Quickview")

st.title("Gravitational Wave Quickview (L1 & V1)")
st.markdown("Enter a GPS time to view LIGO/Virgo strain data ±30 s around it.")

@st.cache_data(max_entries=5)
def load_gw(t0, detector, fs=4096, pad=30):
    return TimeSeries.fetch_open_data(detector, t0 - pad, t0 + pad,
                                      sample_rate=fs, cache=False)

detectors = ["L1", "V1"]

# Sidebar controls
st.sidebar.header("Input Parameters")
t0 = float(st.sidebar.text_input("GPS Time", "1126259462.4"))
fs = st.sidebar.selectbox("Sample rate (Hz)", [4096, 16384], index=0)
dtboth = st.sidebar.slider("Time range for plots (s)", 0.2, 8.0, 1.0)
dt = dtboth / 2.0
whiten = st.sidebar.checkbox("Whiten", value=True)
freqrange = st.sidebar.slider("Band-pass range (Hz)", 10, 2000, (30, 400))
vmax = st.sidebar.slider("Spectrogram colorbar max", 10, 500, 25)
qcenter = st.sidebar.slider("Q-value", 5, 120, 5)
qrange = (int(qcenter * 0.8), int(qcenter * 1.2))

st.sidebar.markdown("---")
st.sidebar.write("Will fetch data ±30 s around the GPS time for L1 and V1.")

strain_load_state = st.text("Loading data...")
strains = {}

for det in detectors:
    try:
        strains[det] = load_gw(t0, det, fs)
    except Exception as e:
        st.warning(f"{det} data unavailable: {e}")

strain_load_state.text("Data loaded.")
cropstart, cropend = t0 - dt, t0 + dt

# === Plot per detector ===
for det, strain_data in strains.items():
    st.header(f"Detector {det}")
    strain = deepcopy(strain_data)

    # --- Raw strain ---
    with _lock:
        fig_raw = strain.crop(cropstart, cropend).plot()
        st.pyplot(fig_raw, clear_figure=True)

    # --- Whitened + Bandpass ---
    if whiten:
        data = strain.whiten().bandpass(freqrange[0], freqrange[1])
    else:
        data = strain.bandpass(freqrange[0], freqrange[1])
    cropped = data.crop(cropstart, cropend)

    st.subheader("Whitened & Band-passed Data")
    with _lock:
        fig_bp = cropped.plot()
        st.pyplot(fig_bp, clear_figure=True)

    # --- CSV download ---
    df = pd.DataFrame({"Time": cropped.times.value, "Strain": cropped.value})
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    fn = f"{det}-STRAIN-{int(cropstart)}-{int(cropend-cropstart)}.csv"
    href = f'<a href="data:file/csv;base64,{b64}" download="{fn}">Download data (CSV)</a>'
    st.markdown(href, unsafe_allow_html=True)

    # --- Spectrogram ---
    st.subheader("Spectrogram")
    q = strain.q_transform(outseg=(t0 - dt, t0 + dt), qrange=qrange)
    with _lock:
        fig_q = q.plot()
        ax = fig_q.gca()
        fig_q.colorbar(label="Normalized energy", vmax=vmax, vmin=0)
        ax.grid(False)
        ax.set_yscale("log")
        ax.set_ylim(bottom=15)
        st.pyplot(fig_q, clear_figure=True)

st.markdown("""
---
Data courtesy of the Gravitational-Wave Open Science Center ([GWOSC](https://gwosc.org)).
""")
