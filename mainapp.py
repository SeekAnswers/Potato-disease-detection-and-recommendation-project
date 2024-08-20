import streamlit as st
import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
from PIL import Image

# Load the trained model (ensure you use the correct path)
model = models.load_model(r'C:\Users\kccha\OneDrive\Desktop\Programming\Potato disease classification\potato_disease_model.keras')

# Get the class names
classname = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Pre-process the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((256, 256))  # 256x256 was used during training
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

# Make predictions by integrating prediction function into the Streamlit app
def predict(model, img_array):
    prediction = model.predict(img_array)
    predicted_class = classname[np.argmax(prediction[0])]
    confidence = round(100 * np.max(prediction[0]), 2)
    return predicted_class, confidence

# Initialize session state variables
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'show_recommendations' not in st.session_state:
    st.session_state.show_recommendations = False

# Build the Streamlit interface
st.header('Potato Disease Prediction/Diagnosis Assistant/System')

uploaded_image = st.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    img_array = preprocess_image(uploaded_image)

    st.subheader('Likely Disease')
    if st.button('Predict'):
        predicted_class, confidence = predict(model, img_array)
        st.session_state.prediction_made = True
        st.session_state.predicted_class = predicted_class
        st.session_state.confidence = confidence
        st.session_state.show_recommendations = False  # Reset the recommendation state

    # Display the prediction results
    if st.session_state.prediction_made:
        st.write(f'Predicted Class: {st.session_state.predicted_class}')
        st.write(f'Confidence: {st.session_state.confidence}%')

        # Toggle button to show/hide recommendations
        if st.button('Show/Hide Recommendations'):
            st.session_state.show_recommendations = not st.session_state.show_recommendations

    # Display the recommendations if the user has clicked "Show Recommendations"
    if st.session_state.show_recommendations:
        #st.subheader('Recommendations')
        if st.session_state.predicted_class == 'Potato___Early_blight':
            st.write('### Recommendations for Potato Early Blight')
            st.write("""
            1. **Crop Rotation:** Rotate potatoes with non-host crops (e.g., cereals) every 2-3 years.
            2. **Field Hygiene:** Remove and destroy all infected plant debris after harvest.
            3. **Plant Spacing:** Space plants adequately to improve air circulation.
            4. **Irrigation Management:** Avoid overhead watering; use drip irrigation instead.
            5. **Resistant Varieties:** Plant potato varieties resistant or tolerant to Early Blight.
            6. **Fungicidal Treatments:** 
               - Preventative Sprays: Apply fungicides like Chlorothalonil, Mancozeb, or Azoxystrobin.
               - Rotate fungicides each season to prevent resistance.
            7. **Balanced Nutrition:** Ensure adequate potassium and phosphorus in the soil.
            8. **Regular Monitoring:** Inspect plants regularly for early signs of disease.
            9. **Biological Control:** Use biocontrol agents like Trichoderma spp. to suppress the disease.
            """)
        elif st.session_state.predicted_class == 'Potato___Late_blight':
            st.write('### Recommendation for Potato Late Blight')
            st.write("""
            1. **Use Disease-Free Seed Potatoes:** Always plant certified, disease-free seed potatoes to prevent the introduction of Late Blight into your field.
            2. **Select Resistant Varieties:** Choose potato varieties that are known to be resistant or tolerant to Late Blight.
            3. **Optimize Planting Time:** Plant early in the season to avoid peak periods of Late Blight infection, usually during warm, humid conditions.
            4. **Ensure Proper Plant Spacing:** Space plants adequately to improve air circulation and reduce humidity around the plants, which helps in reducing the spread of the disease.
            5. **Irrigation Management:** Avoid overhead irrigation to keep foliage dry. Prefer drip irrigation to minimize leaf wetness.
            6. **Remove Infected Plants Immediately:** Inspect your fields regularly and remove any infected plants as soon as they are detected to prevent the spread of the disease.
            7. **Implement Crop Rotation:** Rotate potatoes with non-host crops (e.g., cereals or legumes) for at least 2-3 years to reduce the persistence of the pathogen in the soil.
            8. **Apply Preventative Fungicides:** Use fungicides as a preventive measure, especially during favorable conditions for Late Blight (wet and cool weather).
               - Common fungicides include Mancozeb, Chlorothalonil, or Copper-based products. Apply according to the manufacturer's instructions.
            9. **Proper Field Hygiene:** After harvest, remove and destroy all potato plant debris, including volunteer plants, as they can harbor the pathogen over the winter.
            10. **Monitor Weather Conditions:** Keep track of weather forecasts, especially for rain and high humidity, as these conditions are conducive to Late Blight outbreaks. Adjust your management practices accordingly.
            11. **Use Mulch:** Apply organic mulch around plants to reduce soil moisture evaporation and prevent spores from splashing onto the foliage during rainfall.
            12. **Avoid Excessive Nitrogen Fertilization:** Excessive nitrogen can promote lush foliage, which is more susceptible to Late Blight. Balance your fertilizer application to meet the plant's needs without promoting excessive growth.
            """)
        elif st.session_state.predicted_class == 'Potato___healthy':
            st.write('### Recommendation for Healthy Potato Plants')
            st.write('Keep up good crop maintenance to ensure the continued health of your plants.')
