import streamlit as st
import pandas as pd
import numpy as np
import base64, requests
import glob as gl
from PIL import Image
import os, warnings
warnings.filterwarnings("ignore")
from pdf_extract.config import global_config as glob
from pdf_extract.utils import utils

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
        tabs = st.tabs(["Data"])
        tab_data = tabs[0]

    with st.sidebar:
        st.image(os.path.join(glob.UC_CODE_DIR,'templates','agcs_banner.png'), use_column_width=True)
        uploaded_file = st.file_uploader("Upload file:", type = ["pdf"])    # returns byte object

    data_orig, my_resp = None, None
    if uploaded_file is not None:
        try:
            data_orig = utils.extract_pdf_data(uploaded_file)
            body = {"text": data_orig.text.tolist(), "fname" : data_orig.fname.tolist()}

            with tab_data:
                dataset = st.expander(label = "Display raw text")
                with dataset:            
                    st.table(data_orig)

        except Exception as ex:
            st.error("Invalid File"); print(ex)

        with st.sidebar:
            #submitted = st.button('Run analysis', key='my_button', on_click = widget_callback)    # boolean 
            st.text(" ")
            if st.button('Get model decision', key='predict'):    # no callback needed here
                response = requests.post(f"http://{glob.UC_APP_CONNECTION}:{glob.UC_PORT}/predict", json=body)
                print(response)
                my_resp = response.json()

        if my_resp is not None:
            with tab_data:
                st.success(f"Model prediction: {my_resp['prediction'][0]} ({np.round(my_resp['proba of pos.'][0], 3)})")
                print(my_resp)
                    

###########
# Run app:
###########
main()


