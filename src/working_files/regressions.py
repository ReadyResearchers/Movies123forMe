import streamlit as st
from src.working_files import etl

import plotly.express as px


def regression():
    """Run linear regression on opus data."""
    opus = etl.clean_data()[0]
    st.write(opus.head())
    col = opus.columns
    columns = {1: opus.columns[0], 2: opus.columns[1], 3: opus.columns[2],
        4: opus.columns[3], 5: opus.columns[4], 6: opus.columns[5],
        7: opus.columns[6], 8: opus.columns[7], 9: opus.columns[8],
        10: opus.columns[9], 11: opus.columns[10], 12: opus.columns[11],
        13: opus.columns[12], 14: opus.columns[13], 15: opus.columns[14],
        16: opus.columns[15], 17: opus.columns[16], 18: opus.columns[17],
        19: opus.columns[18]}
    category = st.selectbox("Select a category to analyze", list(columns.keys()), format_func=lambda x: columns[x])
    x_val = st.selectbox("X-value: ", col[:])
    y_val = st.selectbox("Y-val: ", col[:8])
    if x_val == y_val:
        x_val = opus.columns[0]
        y_val = opus.columns[1]

    # scatterplot chart
    fig = px.scatter(opus, x = x_val, y = y_val, color = columns[category], 
                    trendline = 'ols', title = f"Scatterplot of {y_val} on {x_val}")
    fig.data[1].line.color = 'red'
    fig.update_xaxes(rangeslider_visible = True)
    st.plotly_chart(fig)
    res = px.get_trendline_results(fig)
    trendline = res['px_fit_results'].iloc[0]
    st.write(trendline.summary())
    # pie chart
    fig1 = px.pie(opus, values = x_val, names = y_val, 
                title = f'Percentage of movie "{y_val}" in "{x_val}"')
    fig1.update_traces(textinfo="percent+value")
    st.plotly_chart(fig1)

    # bar chart
    fig2 = px.bar(opus, x = x_val, y = y_val, color = columns[category],
    title = f'Bar chart of {x_val} on {y_val}')
    st.plotly_chart(fig2)

    #might need to work on this to get the func to work
    fig3 = px.bar(opus, x = x_val, color = columns[category],
    title = f'Bar chart of the count of {x_val}')
    st.plotly_chart(fig3)

regression()
#os.remove(html_fname)