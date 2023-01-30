from src.working_files import etl
import streamlit as st
import plotly.express as px

def data_norm():
    names = {'Opus': 0}
    data = st.selectbox("Choose Dataset: ", names.keys())
    if data == 'Opus':
        columns = ['production_budget', 'domestic_box_office', 'international_box_office']
    else:
        columns = []
    index = st.selectbox(f"Select index to use with normalizing the data: ", etl.clean_data()[names[data]].columns)
    "Max Scaling Data Normalization"
    # max scaling
    # get absolute value of max value and divide all entries by it
    max = etl.clean_data()[names[data]].copy()
    st.write(max)
    max = max.set_index(str(index))
    max[columns] = max[columns] / max[columns].abs().max()
    plot = px.bar(max[columns])
    st.plotly_chart(plot)
    # min max scaling
    # subtract the minimum value from all entries and then divide by the range
    "Min Max Data Normalization"
    min_max = etl.clean_data()[names[data]].copy()
    min_max = min_max.set_index(index)
    min_max[columns] = (min_max[columns] - min_max[columns].min()) / (min_max[columns].max() - min_max[columns].min())
    st.write(min_max)
    plot2 = px.bar(min_max[columns])
    st.plotly_chart(plot2)
    # z score method (standardization)
    # subtract the mean from the rows and then divide by the standard deviation
    # this will tell us how many std away a value is from the mean
    "Z Score (Standardization) Data Normalization"
    z = etl.clean_data()[names[data]].copy()
    z = z.set_index(index)  
    z[columns] = (z[columns] - z[columns].mean()) / z[columns].std()
    st.write(z)
    plot3 = px.bar(z[columns])
    st.plotly_chart(plot3)

data_norm()
text_analysis()