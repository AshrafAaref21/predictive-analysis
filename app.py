import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import plotly.express as px

from utils import to_excel, fill_null, get_classifier, download_model

# from sklearn.preprocessing import LabelEncoder
matplotlib.use('Agg')


st.set_page_config(
    page_title='Auto Data',
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state='expanded'
)
_, cl, _ = st.columns([1, 3, 1])
with cl:
    st.title('Data Automation By ENG/ John')
st.divider()

image = Image.open('data_science.png')
st.image(image, use_column_width=True)


def main(df=''):

    activities = ['EDA', 'Visualization', 'Classifier Model', 'Help']

    option = st.sidebar.selectbox('Select The Activity Option', activities)

    cl1, cl2 = st.columns(2)
    with cl1:
        data_type = st.selectbox('Select The criteria', [
                                 'File Upload', 'Input URL'])
    if data_type == 'File Upload':
        with cl2:
            data_uploader = st.file_uploader(
                "Upload datasets", type=['csv', 'xlsx'])
        if data_uploader is not None:
            if data_uploader.name[-3:] == 'csv':
                df = pd.read_csv(data_uploader)
                count_of_null = df.copy().isnull().sum().sum()
            else:
                df = pd.read_excel(data_uploader)

            st.success('Your Data Successfully Uploaded')
            st.dataframe(df.head(30), use_container_width=True)
    else:
        with cl2:
            url = st.text_input('Input URL')
            try:
                df = pd.read_csv(
                    url)
                count_of_null = df.copy().isnull().sum().sum()
            except:
                st.error('It\'s not a valid URL')
        if type(df) != str:
            st.success('Your Data Successfully Uploaded')
            st.dataframe(df.head(30), use_container_width=True)

    st.divider()

    if type(df) != str:

        if option == 'EDA':
            st.subheader('Exploratory Data Analysis')
            if st.checkbox("Display shape"):
                st.write(df.shape)

            if st.checkbox("Display columns"):
                all_columns = df.columns.to_list()
                st.write(all_columns)

            if st.checkbox('Select Multiple columns'):
                selected_columns = st.multiselect(
                    "Select Preferred columns", df.columns)
                df1 = df[selected_columns]
                st.dataframe(df1)

            if st.checkbox("Display summary"):
                st.dataframe(df.describe().T, use_container_width=True)

            if st.checkbox("Null Values"):
                st.dataframe(df.isnull().sum().rename(
                    'Count Of Null Values'), use_container_width=True)

            if st.checkbox('Data Types'):
                st.dataframe(df.dtypes.rename(
                    'Data Type'), use_container_width=True)

            if st.checkbox('Display Correlation'):
                st.dataframe(df.corr(numeric_only=True),
                             use_container_width=True)

            if st.checkbox('Treat The Data'):

                df = fill_null(df)
                st.download_button(label='📥 Download The Excel Sheet After beign treated',
                                   data=to_excel(df),
                                   type='primary',
                                   file_name=r'Seer.xlsx',
                                   use_container_width=True,
                                   disabled=count_of_null == df.isnull().sum().sum()
                                   )
            st.divider()

        elif option == 'Visualization':
            st.subheader('Interactive Data Visualization')
            st.divider()

            cols_box = st.checkbox('Select Multiple Columns to plot')
            if cols_box:
                selected_columns = st.multiselect(
                    "Select Preferred columns", df.columns)
                df1 = df[selected_columns]
                st.dataframe(df1)

            if st.checkbox("Display Heatmap"):
                fig = plt.figure()
                if cols_box:
                    sns.heatmap(df1.corr(numeric_only=True), vmax=1,
                                square=True, annot=True, cmap='viridis')
                else:
                    sns.heatmap(df.corr(numeric_only=True), vmax=1,
                                square=True, annot=True, cmap='viridis')
                st.pyplot(fig)

            if st.checkbox("Display Pairplot"):
                if cols_box:
                    fig = sns.pairplot(df1, kind='kde')
                else:
                    fig = sns.pairplot(df, kind='kde')
                st.pyplot(fig)

            if st.checkbox("Pie Chart"):
                all_columns = df.columns.to_list()
                pie_column = st.selectbox(
                    "Select a column, NB: Select Target column", all_columns)
                if df[pie_column].dtype in [float, int]:
                    st.error(
                        f'Feature ({pie_column}) is not a categorical feature, Try to use another Feature')
                else:
                    fig = plt.figure()
                    df[pie_column].value_counts().plot.pie(
                        autopct="%1.1f%%", label='')
                    st.pyplot(fig)

            if st.checkbox("Interactive Scatter Plot"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    x = st.selectbox('X axis', df.columns)
                with col2:
                    y = st.selectbox(
                        'Y axis', [None, *[i for i in list(df.columns) if i != x]])
                with col3:
                    c = st.selectbox('Clustering Color Feature', [
                        None, *[i for i in list(df.columns) if i not in [x, y]]])
                with col4:
                    s = st.selectbox('Clustering Size Feature', [
                        None, *[i for i in list(df.columns) if i not in [x, y, c]]])
                try:
                    fig = px.scatter(
                        df,
                        x=x,
                        y=y,
                        size=s,
                        color=c,
                        color_continuous_scale="reds"
                        # hover_name="country",
                        # log_x=True,
                        # size_max=30,
                    )
                    if y == None and c == None and s == None:
                        pass
                    else:
                        st.plotly_chart(fig, theme="streamlit",
                                        use_container_width=True)
                except:
                    st.error(
                        'You have a feature with null values, You have to treat the data and upload it, Then try again. ')

        elif option == 'Classifier Model':
            st.subheader('Predictive Model')
            st.divider()
            classifier_name = st.sidebar.selectbox(
                'Select classifier', ('KNN', 'SVM', 'LR', 'naive_bayes', 'DecisionTree'))

            col1, col2 = st.columns(2)

            with col1:
                y_label = st.selectbox('Select The Target Column', df.columns)

            with col2:
                x_label = st.multiselect('Select The Feature Columns', [
                    i for i in df.columns if i != y_label])

            X = df[x_label]
            y = df[y_label]

            clf = get_classifier(classifier_name)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            if len(x_label) > 0:
                if df[y_label].isnull().sum() == 0:

                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    st.divider()
                    st.write("Classifier Accuracy Score: ",
                             f'**{np.round(100 * accuracy_score(y_test, y_pred),2)}** %'
                             )

                    download_model(clf)
                else:
                    st.error(
                        'Your Target Feature has null values, You have to treat your data first')

            else:
                st.write('\n')
                st.warning('Kindly, Select The Target and Feature Columns')
        else:
            st.info("""
        
                    Hello, It's a Product to help you discover your data and get advanced insights of it. 
                    \n
                    You can download your csv, xlsx data or provide us with a url of it and we will display your data and help you with analysis reports, visualizations, and predictive classification model

                    """)


if __name__ == '__main__':
    main()
