import streamlit as st
import ab_page
import clv_page
import upload_page

st.set_page_config(layout="wide")

page = st.sidebar.selectbox('Select page', [ 'Select Page', 'CLV Page', 'Experiment Planning','Automated Experiment Analysis'])


if page == 'Select Page':
    pass
elif page == 'Automated Experiment Analysis':
    upload_page.upload_page()
elif page == 'Experiment Planning':
    ab_page.AB_page()
elif page == "CLV Page":
    clv_page.clv_page()
