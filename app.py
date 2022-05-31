import streamlit as st 
from fastai.vision.all import *
import plotly.express as px
import pathlib
import platform

plt = platform.system()
if plt=='Linux': pathlib.WindowsPath= pathlib.PosixPath


st.title('Classification of Boat,Car and Airplane')

file = st.file_uploader('Upload picture', type=['jpeg','png','gif','svg'])
if file:
    st.image(file)

    img = PILImage.create(file)

    model = load_learner('transport_class_model.pkl')

    pred,pred_id,probs = model.predict(img)
    st.success(f"Prediction: {pred}")
    st.info(f"Probability: {probs[pred_id]*100:.1f}%")

    fig = px.bar(x=probs*100,y= model.dls.vocab)
    st.plotly_chart(fig)

