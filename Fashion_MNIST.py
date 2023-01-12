import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.preprocessing import StandardScaler
import keras

model = keras.models.load_model('models/fashion_model.h5')
Indicator_mnist = pd.read_csv('Fashion_MNIST/Indicator_mnist.csv')

st.title('Model weight:')
st.image('Fashion_MNIST/fashion_model.png', channels="BGR")

st.title('Model evaluation indicators:')
st.write('Accuracy')
st.line_chart(Indicator_mnist[["acc", "val_acc"]])
st.write('loss')
st.line_chart(Indicator_mnist[["loss", "val_loss"]])

# Upload pictures and display
uploaded_file = st.file_uploader("Upload a picture")

if uploaded_file is not None:
    # show image
    st.image(uploaded_file, channels="BGR")
    scaler = StandardScaler()
    I = Image.open(uploaded_file)
    I.save('test.jpg')
    im = Image.open('test.jpg')
    im = im.resize((28, 28), Image.ANTIALIAS)
    L = im.convert('L')
    im2 = np.array(L)
    im2 = np.array([255 - im2])
    im3 = scaler.fit_transform(im2.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
    y = model.predict(im3)
    class_names = ['T-shirt', 'trousers', 'pullover', 'dress', 'coat',
                   'Sandals', 'Shirts', 'Sneakers', 'Bags', 'Boots']
    st.write('Weighted score：')
    st.dataframe(pd.DataFrame(y[0], index=class_names, columns=["weighted_score"]))
    st.write('The predicted picture is：')
    st.title(class_names[np.argmax(y[0])])
