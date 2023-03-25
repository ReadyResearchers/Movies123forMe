import streamlit as st
import plotly.express as px
import warnings
import base64

from pages import clean_data
warnings.filterwarnings('ignore')

st.markdown("<style>h1 {text-align: center;}</style><h1>Welcome to the Movie Analysis Experience 🎈</h1>", unsafe_allow_html=True)
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

st.markdown("<h3>The Movie Analysis Experience 🎈</h3>", unsafe_allow_html=True)
st.sidebar.markdown("# Page 1 🎈")

opus = clean_data.clean_data()[0]

# finding the mean of a column
dataset = 'Opus'
if dataset == 'Opus':
    col = opus.columns
    columns = {1: opus.columns[0], 2: opus.columns[1], 3: opus.columns[2],
        4: opus.columns[3], 5: opus.columns[4], 6: opus.columns[5],
        7: opus.columns[6], 8: opus.columns[7], 9: opus.columns[8],
        10: opus.columns[9], 11: opus.columns[10], 12: opus.columns[11],
        13: opus.columns[12], 14: opus.columns[13], 15: opus.columns[14],
        16: opus.columns[15], 17: opus.columns[16], 18: opus.columns[17],
        19: opus.columns[18], 20: opus.columns[19]}
    column = st.sidebar.selectbox("Please specify a column to find the mean: ", list(columns.keys()), format_func=lambda x: columns[x])
    type = st.sidebar.selectbox("Please specify a column to group by: ", list(columns.keys()), format_func=lambda x: columns[x])
    # c = columns[column]
    if type == column:
        type = 5
        column = 4
    opus.groupby(columns[type])[columns[column]].transform('mean')

    # find out how many columns/rows are in the data
    #st.write(opus.shape)

    # find out information about the data
    #st.write(opus.info())

    # dropping identifier column
    opus = opus.drop(columns = ['movie_odid'], axis=1)

    
    # analysis between count of entries of the columns
    opus = opus.groupby(columns[type])[columns[column]].size().reset_index()
    st.plotly_chart(px.bar(data_frame=opus, x=columns[type], y=columns[column], color=columns[type], barmode="group",
    title=f"Analysis of the Count of {columns[column]} compared to the {columns[type]}"
    ))

    st.plotly_chart(px.scatter(data_frame = opus, x = columns[type], y = columns[column], color = columns[type]))
    st.plotly_chart(px.funnel_area(names=opus[columns[type]].value_counts().index[0:12], 
    values=opus[columns[type]].value_counts().values[0:12], opacity = 0.7))

if dataset == 'Choose My Own Data':
    st.write("Coming soon...")