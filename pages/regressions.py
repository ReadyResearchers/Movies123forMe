import streamlit as st
from pages import clean_data
import base64
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px


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
st.markdown("<h3>The Data Regression Experience 🎈</h3>", unsafe_allow_html=True)
st.sidebar.write(" Page 5 🎈")

def regression():
    """Run linear regression on opus data."""
    opus = clean_data.clean_data()[0]
    st.write(opus.head())
    # finding the mean production budget of a successful movie
    #for i in opus['movie_success']:
        #if i == 1:
            #print(opus['production_budget'].mean())
    # col = opus.columns
    use_demo = st.sidebar.checkbox('Use Demo of Regression:')
    if use_demo:
        # scatterplot chart
        fig = px.scatter(opus, x = 'genre_Black Comedy', y = 'movie_success', color = 'rating', 
                        trendline = 'ols', title = f"Scatterplot of 'movie_success' on 'genre_Black Comedy'")
        fig.data[1].line.color = 'red'
        fig.update_xaxes(rangeslider_visible = True)
        st.plotly_chart(fig)
        # pie chart
        fig1 = px.pie(opus, values = 'genre_Black Comedy', names = 'movie_success', 
                    title = f"Percentage of 'movie_success' from 'genre_Black Comedy'")
        fig1.update_traces(textinfo="percent+value")
        st.plotly_chart(fig1)

        # bar chart
        fig2 = px.bar(opus, x = 'genre_Black Comedy', y = 'movie_success', color = 'rating',
        title = f"Bar chart of 'movie_success' on 'genre_Black Comedy'")
        st.plotly_chart(fig2)

        #might need to work on this to get the func to work
        fig3 = px.bar(opus, 'genre_Black Comedy', color = 'rating',
        title = f"Bar chart of the count of 'genre_Black Comedy'")
        st.plotly_chart(fig3)

        fig4, ax = plt.subplots()
        sns.regplot(x='genre_Black Comedy', y='movie_success', data=opus, logistic=True, ci=None, ax=ax,
        scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})
        st.write(fig4)

        show_stat = st.sidebar.checkbox('Show Summary statistics:')
        if show_stat:
            res = px.get_trendline_results(fig)
            trendline = res['px_fit_results'].iloc[0]
            st.write(trendline.summary())
    else:
        columns1= {1: opus.columns[0], 2: opus.columns[1], 3: opus.columns[2],
            4: opus.columns[3], 5: opus.columns[4], 6: opus.columns[5],
            7: opus.columns[6], 8: opus.columns[7], 9: opus.columns[8],
            10: opus.columns[9], 11: opus.columns[10], 12: opus.columns[11],
            13: opus.columns[12], 14: opus.columns[13], 15: opus.columns[14],
            16: opus.columns[15], 17: opus.columns[16], 18: opus.columns[17],
            19: opus.columns[18]}
        columns_num = {1: opus.columns[1], 2: opus.columns[2], 3: opus.columns[3], 
            4: opus.columns[4], 5: opus.columns[5], 6: opus.columns[6], 7: opus.columns[10],
            8: opus.columns[11],
            9: opus.columns[12], 10: opus.columns[13], 11: opus.columns[14],
            12: opus.columns[15], 13: opus.columns[16], 14: opus.columns[17],
            15: opus.columns[18]}
        category = st.sidebar.selectbox("Select a category to analyze", list(columns1.keys()), format_func=lambda x: columns1[x])
        x_val = st.sidebar.selectbox("X-value: ", list(columns_num.keys()), format_func=lambda x: columns_num[x])
        y_val = st.sidebar.selectbox("Y-val: ", list(columns_num.keys()), format_func=lambda x: columns_num[x])
        if x_val == y_val:
            x_val = 3
            y_val = 11
            category = 11

        # scatterplot chart
        fig = px.scatter(opus, x = columns_num[x_val], y = columns_num[y_val], color = columns1[category], 
                        trendline = 'ols', title = f"Scatterplot of {columns_num[y_val]} on {columns_num[x_val]}")
        fig.data[1].line.color = 'red'
        fig.update_xaxes(rangeslider_visible = True)
        st.plotly_chart(fig)
        # pie chart
        fig1 = px.pie(opus, values = columns_num[x_val], names = columns_num[y_val], 
                    title = f'Percentage of movie "{columns_num[y_val]}" in "{columns_num[x_val]}"')
        fig1.update_traces(textinfo="percent+value")
        st.plotly_chart(fig1)

        # bar chart
        fig2 = px.bar(opus, x = columns_num[x_val], y = columns_num[y_val], color = columns1[category],
        title = f'Bar chart of {columns_num[x_val]} on {columns_num[y_val]}')
        st.plotly_chart(fig2)

        #might need to work on this to get the func to work
        fig3 = px.bar(opus, x = columns_num[x_val], color = columns1[category],
        title = f'Bar chart of the count of {columns_num[x_val]}')
        st.plotly_chart(fig3)

        fig4, ax = plt.subplots()
        sns.regplot(x=columns_num[x_val], y=columns_num[y_val], data=opus, logistic=True, ci=None,
        ax=ax, scatter_kws={'color': 'black'}, line_kws={'color': 'red'})
        st.write(fig4)

        show_stat = st.sidebar.checkbox('Show Summary statistics:')
        if show_stat:
            res = px.get_trendline_results(fig)
            trendline = res['px_fit_results'].iloc[0]
            st.write(trendline.summary())

regression()
#os.remove(html_fname)