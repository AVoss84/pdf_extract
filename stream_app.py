import streamlit as st
import pandas as pd
import numpy as np
import base64, requests
import glob as gl
from PIL import Image
import os, warnings
warnings.filterwarnings("ignore")
from src.pdf_extract.config import global_config as glob
from src.pdf_extract.utils import utils

#--------------------------------------------------------
# Set Page name and icon, Layout and sidebar expanded
#--------------------------------------------------------
img = Image.open(os.path.join(glob.UC_CODE_DIR,'templates','allianz_logo.jpg'))    # page name icon
st.set_page_config(page_title='Anomaly Report Creator', page_icon=img, layout="wide", initial_sidebar_state='expanded')
#----------------------------------------------------------------------------------------------------------------------


######################################################################################
################################# Start App   ########################################
######################################################################################

def main():

    header = st.container()

    with header:
        tabs = st.tabs(["Data", "Predictions"])
        tab_data, tab_pred = tabs[0], tabs[1]

    with st.sidebar:
        st.image(os.path.join(glob.UC_CODE_DIR,'templates','agcs_banner.png'), use_column_width=True)
        path = st.file_uploader("Upload file(s):", type = ["pdf"], accept_multiple_files=True)    # returns byte object

    data_orig, my_resp, df_resp = None, None, None
    if path is not None:
        try:
            query = pd.DataFrame(columns=['File name','Text'])
            for z, uploaded_file in enumerate(path):
                text, fname = utils.extract_pdf_data(uploaded_file)
                query.loc[z] = [fname, text]

            # Prepare body for http POST request
            body = {"text": query['Text'].tolist(), "fname" : query['File name'].tolist()}

            with tab_data:
                dataset = st.expander(label = "Display raw text")
                with dataset:            
                    st.table(query)

        except Exception as ex:
            st.error("Invalid File"); print(ex)

        with st.sidebar:
            st.text(" ")
            if st.button('Get model decision', key='predict'):    # no callback needed here
                try:
                    response = requests.post(f"http://{glob.UC_APP_CONNECTION}:{glob.UC_PORT}/predict", json=body)
                    print(response)
                    my_resp = response.json()
                    df_resp = pd.DataFrame(my_resp).sort_values(by="proba of pos.", ascending=False)
                    df_resp.rename(columns={'prediction': 'Prediction', 'proba of pos.' : 'Probability of positive', 'fname': 'File name'}, inplace=True)
                except Exception as ex:
                    print(ex)

        if my_resp is not None:
            with tab_data:
                st.success(f"Check model predictions.")
                print(my_resp)
                    
            with tab_pred:
                m_out = st.expander(label = "Display model output")
                with m_out:            
                    st.table(df_resp)

###########
# Run app:
###########
main()

# streamlit run stream_app.py
