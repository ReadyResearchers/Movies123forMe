import matplotlib.pyplot as plt
import streamlit as st
from fpdf import FPDF
import base64
import numpy as np
from tempfile import NamedTemporaryFile

import regressions

def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download Plots</a>'


opus = regressions.linear_regression()

figs = []

for col in opus.columns:
    fig, ax = plt.subplots()
    ax.plot(opus[col])
    st.pyplot(fig)
    figs.append(fig)

export_as_pdf = st.button("Export Plots")

if export_as_pdf:
    pdf = FPDF()
    for fig in figs:
        pdf.add_page()
        with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                fig.savefig(tmpfile.name)
                pdf.image(tmpfile.name, 10, 10, 200, 100)
    html = create_download_link(pdf.output(dest="S").encode("latin-1"), "plots_file")
    st.markdown(html, unsafe_allow_html=True)
