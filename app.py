import streamlit as st
import cv2
import numpy as np
from skimage import feature
from imutils import paths
import pickle
import pandas as pd
import joblib
from keras.models import load_model

# Load the trained model
model_path = "spiralModel.pkl"
with open(model_path, "rb") as model_file:
    spiralModel = pickle.load(model_file)

def quantify_image(image):
    features = feature.hog(image, orientations=9,
                           pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                           transform_sqrt=True, block_norm="L1")
    return features

def predict(image):
    # Preprocess the input image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))
    image = cv2.threshold(image, 0, 255,
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    features = quantify_image(image)

    # Make predictions using the trained model
    prediction = spiralModel["classifier"].predict([features])[0]
    return prediction

def home_page():
    st.title("NeuroAid")

    st.write(
        """
        Neurodegenerative diseases are a group of disorders characterized by the progressive degeneration of the structure and function of the nervous system. These diseases primarily affect neurons, leading to problems with movement, cognition, and various other functions.
        """
    )

    # Section 1: Introduction
    st.subheader("Introduction")
    st.write(
        """
        Early detection and awareness are crucial for managing these conditions effectively. In this app, we focus on using machine learning to identify potential signs of neurodegenerative diseases from spiral drawings.
        """
    )

    # Section 2: Common Neurodegenerative Diseases
    st.subheader("Common Neurodegenerative Diseases")
    st.write(
        """
        - Alzheimer's Disease
        - Parkinson's Disease
        - Amyotrophic Lateral Sclerosis (ALS)
        - Huntington's Disease
        """
    )

    # Section 3: Importance of Early Detection
    st.subheader("Importance of Early Detection")
    st.write(
        """
        Learn more about the importance of early detection and raise awareness about neurodegenerative diseases. Explore the other pages for resources and Parkinson's disease prediction.
        """
    )

def resources_page():
    st.title("Resources - Neurodegenerative Disease Resources")

    st.write(
        """
        Here, you can find resources related to neurodegenerative diseases, including information, support groups, and links to relevant organizations.
        """
    )

    # Section 1: Educational Resources
    st.subheader("Educational Resources")
    st.write(
        """
        - [Alzheimer's Association](https://www.alz.org/)
        - [Parkinson's Foundation](https://www.parkinson.org/)
        - [ALS Association](https://www.alsa.org/)
        - [Huntington's Disease Society of America](https://hdsa.org/)
        """
    )

    # Section 2: Support Groups
    st.subheader("Support Groups")
    st.write(
        """
        - [PatientsLikeMe - Neurological Conditions](https://www.patientslikeme.com/)
        - [Smart Patients - Neurological Disorders](https://www.smartpatients.com/)
        """
    )

    # Section 3: Research and Clinical Trials
    st.subheader("Research and Clinical Trials")
    st.write(
        """
        - [ClinicalTrials.gov - Neurodegenerative Diseases](https://clinicaltrials.gov/ct2/home)
        """
    )

def parkinsons_prediction_page():
    st.title("Parkinson's Prediction - Predicting Parkinson's Disease")

    st.write(
        """
        On this page, you can upload a spiral drawing, and the app will predict whether it shows signs of Parkinson's disease based on the trained machine learning model.
        """
    )

    template_image_path = "template.jpg"
    template_image = cv2.imread(template_image_path)
    st.image(template_image, caption="Spiral Template", use_column_width=300)


    # File Upload
    uploaded_file = st.file_uploader("Choose a spiral drawing...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        st.image(image, caption="Uploaded Image", use_column_width=300)

        # Make predictions
        prediction = predict(image)

        if (prediction==0):
            st.write(f"Prediction  Normal: AI model detects no signs of Parkinsons")
        else:
            st.write(f"Prediction Atypical: AI model detects signs of Parkinsons")

def voice_analysis_page():
    st.title('Voice-Based Parkinson\'s Disease Analysis')
    st.sidebar.header('User Input Features')

    data = {
    'avg_fre': [60.0, 180.0, 300.0],  # Add values for other features as needed
    'max_fre': [80.0, 350.0, 620.0],
    'min_fre': [40.0, 155.0, 270.0],

    'var_fre1': [0.00, 0.020, 0.040],

    'var_fre2': [0.00, 0.00, 0.01],

    'var_fre3': [0.00, 0.01, 0.02],

    'var_fre4': [0.00, 0.035, 0.07],

    'var_fre5': [0.00, 0.1, 0.2],

    'var_amp1': [0.0, 0.1, 0.2],
    'var_amp2': [0.0, 0.03, 0.06],
    'var_amp3': [0.0, 0.4, 0.08],
    'var_amp4': [0.0, 0.1, 0.2],
    'var_amp5': [0.0, 0.1, 0.2],

    'NHR': [0.0, 0.20, 0.4],
    'HNR': [6.0, 21, 36.0],
    'RPDE': [0.2, 0.5, 0.8],
    'DFA': [0.4, 0.65, 0.9],
    'spread1': [-9.0, -4.0, -1.0],
    'spread2': [0.00, 0.25, 0.5],
    'D2': [1.3, 2.5, 3.8],
    'PPE': [0.03, 0.31, 0.6],
    'status': [0, 1, 0]  # Assuming 'status' is the target column
    }

    df = pd.DataFrame(data)
    
    # Load the Random Forest model
    model_path = 'modelrf.pkl'  # Update with the correct path to your saved Random Forest model
    with open(model_path, 'rb') as model_file:
        rf_model = pickle.load(model_file)
    
    # Create sliders for user input features
    avg_fre = st.sidebar.slider('Average Frequency (Hz)', float(df['avg_fre'].min()), float(df['avg_fre'].max()), float(df['avg_fre'].mean()))
    max_fre = st.sidebar.slider('Maximum Frequency (Hz)', float(df['max_fre'].min()), float(df['max_fre'].max()), float(df['max_fre'].mean()))
    min_fre = st.sidebar.slider('Minimum Frequency (Hz)', float(df['min_fre'].min()), float(df['min_fre'].max()), float(df['min_fre'].mean()))
    var_fre1 = st.sidebar.slider('Jitter Percentage', float(df['var_fre1'].min()), float(df['var_fre1'].max()), float(df['var_fre1'].mean()))
    var_fre2 = st.sidebar.slider('Jitter Absolute', float(df['var_fre2'].min()), float(df['var_fre2'].max()), float(df['var_fre2'].mean()))
    var_fre3 = st.sidebar.slider('Jitter RAP', float(df['var_fre3'].min()), float(df['var_fre3'].max()), float(df['var_fre3'].mean()))
    var_fre4 = st.sidebar.slider('Jitter PPQ', float(df['var_fre4'].min()), float(df['var_fre4'].max()), float(df['var_fre4'].mean()))
    var_fre5 = st.sidebar.slider('Jitter DDP', float(df['var_fre5'].min()), float(df['var_fre5'].max()), float(df['var_fre5'].mean()))
    var_amp1 = st.sidebar.slider('Shimmer', float(df['var_amp1'].min()), float(df['var_amp1'].max()), float(df['var_amp1'].mean()))
    var_amp2 = st.sidebar.slider('Shimmer (dB)', float(df['var_amp2'].min()), float(df['var_amp2'].max()), float(df['var_amp2'].mean()))
    var_amp3 = st.sidebar.slider('Shimmer APQ3', float(df['var_amp3'].min()), float(df['var_amp3'].max()), float(df['var_amp3'].mean()))
    var_amp4 = st.sidebar.slider('Shimmer APQ5', float(df['var_amp4'].min()), float(df['var_amp4'].max()), float(df['var_amp4'].mean()))
    var_amp5 = st.sidebar.slider('Shimmer DDA', float(df['var_amp5'].min()), float(df['var_amp5'].max()), float(df['var_amp5'].mean()))
    NHR = st.sidebar.slider('NHR', float(df['NHR'].min()), float(df['NHR'].max()), float(df['NHR'].mean()))
    HNR = st.sidebar.slider('HNR', float(df['HNR'].min()), float(df['HNR'].max()), float(df['HNR'].mean()))
    RPDE = st.sidebar.slider('RPDE', float(df['RPDE'].min()), float(df['RPDE'].max()), float(df['RPDE'].mean()))
    DFA = st.sidebar.slider('DFA', float(df['DFA'].min()), float(df['DFA'].max()), float(df['DFA'].mean()))
    spread1 = st.sidebar.slider('Spread1', float(df['spread1'].min()), float(df['spread1'].max()), float(df['spread1'].mean()))
    spread2 = st.sidebar.slider('Spread2', float(df['spread2'].min()), float(df['spread2'].max()), float(df['spread2'].mean()))
    D2 = st.sidebar.slider('D2', float(df['D2'].min()), float(df['D2'].max()), float(df['D2'].mean()))
    PPE = st.sidebar.slider('PPE', float(df['PPE'].min()), float(df['PPE'].max()), float(df['PPE'].mean()))

    user_input = [avg_fre, max_fre, min_fre, var_fre1, var_fre2, var_fre3, var_fre4, var_fre5,
                  var_amp1, var_amp2, var_amp3, var_amp4, var_amp5, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

    # Display user input features
    st.subheader('User Input Features')
    user_input_df = pd.DataFrame(data=[user_input], columns=df.columns[:-1])  # Assuming the last column is the target 'status'
    st.write(user_input_df)

    # Make prediction with the Random Forest model
    prediction = rf_model.predict(user_input_df)
    
    # Display Random Forest model prediction
    st.subheader('Random Forest Model Prediction')
    st.write(f"Random Forest Model: {int(prediction[0])}")

def dementia():
    st.title("Dementia Detection")

    uploaded_file = st.file_uploader("Choose an image for dementia detection...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        st.image(image, caption="Uploaded Image", use_column_width=300)

        # Preprocess the input image manually (you may need to adjust this based on your model's requirements)
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure the color is correct for the model
        processed_image = cv2.resize(processed_image, (224, 224))
        processed_image = processed_image / 255.0  # Normalize pixel values to be in the range [0, 1]

        # Reshape the image to match the model's input shape
        processed_image = processed_image.reshape((1, 224, 224, 3))

        # Load the dementia model using Keras's load_model
        dementia_model_path = "dementia.h5"  # Assuming you saved the model with the .h5 extension
        dementia_model = load_model(dementia_model_path)

        # Make predictions using the loaded model
        prediction = dementia_model.predict(processed_image)[0]

        confidence = prediction[np.argmax(prediction)] * 100  # Confidence in percentage
        if np.argmax(prediction) == 1:
            st.write(f"Prediction: Atypical - Signs of dementia detected with confidence: {confidence:.2f}%")
        else:
            st.write(f"Prediction: Normal - No signs of dementia with confidence: {confidence:.2f}%")


def main():
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Go to", ["Home", "Resources", "Parkinson's Prediction", "Voice-Based Analysis", "Dementia Detection"])

    if selected_page == "Home":
        home_page()
    elif selected_page == "Resources":
        resources_page()
    elif selected_page == "Parkinson's Prediction":
        parkinsons_prediction_page()
    elif selected_page == "Voice-Based Analysis":
        voice_analysis_page()
    elif selected_page == "Dementia Detection":
        dementia()

if __name__ == "__main__":
    main()

