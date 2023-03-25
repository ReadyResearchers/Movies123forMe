import streamlit as st
import base64
from pathlib import Path


def read_markdown_file(markdown_file="main_dashboard.md"):
    return Path(markdown_file).read_text()


def main():
    # displaying the gif header for the landing page
    path='img/movies123forme_header.mp4'
    with open(path, "rb") as f:
        video_content = f.read()
    video_str = f"data:video/mp4;base64,{base64.b64encode(video_content).decode()}"
    st.markdown(f"""
    <center>
        <video style="display: auto; margin: auto; width: 600px;" controls loop autoplay>
            <source src="{video_str}" type="video/mp4">
        </video>
    </center>
    """, unsafe_allow_html=True)
    st.write("---")

    # displaying welcome message and thesis
    st.markdown("""
    **Welcome to the Movies123ForMe website,** *built and hosted on Streamlit!*

    This Streamlit application was built as a senior comprehrensive thesis project, 
    in fulfillment of all requirements brought forth by the Allegheny College 
    Department of Computer Science. To read the thesis that this website was built on, 
    feel free to read/download using the download button below:
    """)

    with open("pages/SeniorThesis.pdf", "rb") as pdf_file:
        PDFbyte = pdf_file.read()

    st.download_button(label="Senior Thesis v2.0.0",
        data=PDFbyte,
        file_name="senior_thesis.pdf",
        mime='application/octet-stream')
    # displaying the descriptions of every page
    intro_markdown = read_markdown_file()
    st.markdown(intro_markdown, unsafe_allow_html=True)