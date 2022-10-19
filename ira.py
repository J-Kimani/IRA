import json
import numpy as np
import pickle
import streamlit as st
import base64


# load saved model
def load_saved_artifacts():
    global __data_columns
    global __model

    with open("./columns.json", 'r') as f:
        __data_columns = json.load(f)["data_columns"]
    
    with open("./insta.pickle", "rb") as f:
        __model = pickle.load(f)


def pred_imp(save, comment, share, like, prof_visit, follow):
    x = np.zeros(len(__data_columns))

    x[0] = save
    x[1] = comment
    x[2] = share
    x[3] = like
    x[4] = prof_visit
    x[5] = follow
    
    imp = np.round(__model.predict([x])[0])

    return imp


@st.experimental_memo

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("Instagram.png")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"]> .main {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid= "stHeader"]{{
background: rgba(0,0,0,0);
}}

</style>
"""


def main():
    # background image
    st.markdown(page_bg_img, unsafe_allow_html=True)


    # Title
    st.title('Instragram Order Prediction')

    # user instructions
    instructions = '<p style="font-family:Courier; color:White; font-size: 20px;">App designed to enable you predict the number of impressions received from your instagram post</p>'
    st.markdown(instructions, unsafe_allow_html=True)


    # user input
    save = st.text_input('Post Saves')
    comment = st.text_input('Comments')
    share = st.text_input('Shares')
    like = st.text_input('Likes')
    prof_visit = st.text_input('Profile Visits')
    follow = st.text_input('Follows')

    # prediction code
    impressions = ''

    # prediction button
    if st.button('Predicted Impressions'):
        impressions = pred_imp(save, comment, share, like, prof_visit, follow)
    
    st.success(impressions)

    # conclusions
    conclusion = '<p style="font-family:Courier; color:White; font-size: 20px;">Negative responses indicate the magnitude of  negative impressions</p>'
    st.markdown(conclusion, unsafe_allow_html=True)



if __name__ == '__main__':
    load_saved_artifacts()
    main()


