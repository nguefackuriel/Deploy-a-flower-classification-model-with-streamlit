import numpy as np
import streamlit as st
import pickle
from GDA import GDA


# Load our mnodel
model_load = pickle.load(open('my_model.sav', 'rb'))

# Define a prediction function

def prediction_model(input_data):
    
    input_data_as_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_array.reshape(1,-1)

    prediction = model_load.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The flower is Iris_setosa'
    elif prediction[0] == 1:
        return 'The flower is Iris-versicolor'
    else:
        return 'The flower is Iris-virginica'


def main():

    # Give a title to the App
    st.title('Iris flower prediction App')

    # Getting the Input from the user
    SepalLengthCm = st.text_input('Length of the sepal in cm')
    SepalWidthCm = st.text_input('Width of the sepal in cm')
    PetalLengthCm = st.text_input('Length of the petal in cm')
    PetalWidthCm = st.text_input('Width of the petal in cm')

    # variable of prediction

    result = ''

    if st.button('Flower test result'):
        result = prediction_model([float(SepalLengthCm),float(SepalWidthCm),float(PetalLengthCm),float(PetalWidthCm)])

    # Display the result
    st.success(result)

if __name__ == '__main__':
    main()