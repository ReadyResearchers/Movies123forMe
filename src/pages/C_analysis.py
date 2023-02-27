import streamlit as st
import plotly.express as px
import warnings

from pages import B_etl
warnings.filterwarnings('ignore')

st.markdown("# Welcome to the Movie Analysis Experience 🎈")
st.sidebar.markdown("# Subpage 1 🎈")

opus = B_etl.clean_data()[0]

# finding the mean of a column

choices = ['Opus', 'Choose My Own Data']
dataset = st.selectbox("Pick a dataset: ", choices)

if dataset == 'Opus':
    col = opus.columns
    columns = {1: opus.columns[0], 2: opus.columns[1], 3: opus.columns[2],
        4: opus.columns[3], 5: opus.columns[4], 6: opus.columns[5],
        7: opus.columns[6], 8: opus.columns[7], 9: opus.columns[8],
        10: opus.columns[9], 11: opus.columns[10], 12: opus.columns[11],
        13: opus.columns[12], 14: opus.columns[13], 15: opus.columns[14],
        16: opus.columns[15], 17: opus.columns[16], 18: opus.columns[17],
        19: opus.columns[18], 20: opus.columns[19]}
    type = st.selectbox("Please specify a column to group by: ", list(columns.keys()), format_func=lambda x: columns[x])
    column = st.selectbox("Please specify a column to find the mean: ", list(columns.keys()), format_func=lambda x: columns[x])
    c = columns[column]
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
    # title=f"Analysis of the Count of {columns[columns]} compared to the {columns[type]}"
    ))

    st.plotly_chart(px.scatter(data_frame = opus, x = columns[type], y = columns[column], color = columns[type]))
    st.plotly_chart(px.funnel_area(names=opus[columns[type]].value_counts().index[0:12], 
    values=opus[columns[type]].value_counts().values[0:12], opacity = 0.7))

if dataset == 'Choose My Own Data':
    st.write("Coming soon...")