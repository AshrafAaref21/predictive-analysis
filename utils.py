from io import BytesIO
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

import pickle
import base64


def download_model(model):
    output_model = pickle.dumps(model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="myfile.pkl">Download Trained Model .pkl File</a>'
    st.markdown(href, unsafe_allow_html=True)


def add_paramters(name):
    params = {}
    if name == 'SVM':
        c = st.sidebar.slider('C', 0.01, 15.0)
        params['C'] = c
    elif name == 'KNN':
        k = st.sidebar.slider('K', 1, 15)
        params['n_neighbors'] = k

    return params


def get_classifier(name):
    clf = None
    params = add_paramters(name)

    if name == 'SVM':
        clf = SVC(**params)
    elif name == 'KNN':
        clf = KNeighborsClassifier(**params)
    elif name == 'LR':
        clf = LogisticRegression(**params)
    elif name == 'naive_bayes':
        clf = GaussianNB(**params)
    elif name == 'DecisionTree':
        clf = DecisionTreeClassifier(**params)

    return clf


def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'})
    worksheet.set_column('A:A', None, format1)
    writer.close()
    processed_data = output.getvalue()
    return processed_data


def fill_null(df, unittest=False, filler=10):

    df_null = df.isnull().sum()
    if unittest:
        return df.fillna(filler)

    if df_null.sum() > 0:

        df_null = df_null[df_null > 0]
        dic = {}

        for i in df_null.index:
            dic[i] = st.selectbox(f'Select filling value of feature {i}', [
                'No-fill', 'mean', 'median', 'mode'])

        btn = st.button('Treat the data', type='primary',
                        use_container_width=True)

        if btn:
            logic = 0
            for i in dic:
                if dic[i] == 'No-fill':
                    logic += 1
                if dic[i] == 'mean':
                    value = df[i].mean()
                elif dic[i] == 'median':
                    value = df[i].median()
                elif dic[i] == 'mode':
                    value = df[i].mode()
                else:
                    value = np.nan
                df[i] = df[i].fillna(value)

            if logic == len(dic):
                st.info('You have to select at least one feature to fill it')
            else:
                st.snow()
                st.success(
                    'The Data has been successfully deleted, You can Download the Treated data now')

    else:
        st.info('Your Data is already clean')

    return df
