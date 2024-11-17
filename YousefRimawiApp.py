import pickle
import pandas as pd
import numpy as np
import streamlit as st


pickle_file = open('assignment.pkl', 'rb')  # Use a separate variable for the file object
model = pickle.load(pickle_file)
pickle_file.close()  # Close the file after loading

# Loading the feature importance DataFrame
pickle_file = open('importance_df.pkl', 'rb')  # Reuse a different file object variable
feature_importance_df = pickle.load(pickle_file)
pickle_file.close()  # Close the file after loading

# Extract the top 10 features
top_features = feature_importance_df['Feature'].head(10).tolist()

# Define the unique values for each feature
unique_values = {
    "odor_n": ["True", "False"],
    "gill_size_n": ["True", "False"],
    "odor_f": ["True", "False"],
    "gill_size_b": ["True", "False"],
    "stalk_surface_below_ring_k": ["True", "False"],
    "spore_print_color_h": ["True", "False"],
    "ring_type_p": ["True", "False"],
    "gill_color_b": ["True", "False"],
    "bruises_f": ["True", "False"],
    "stalk_surface_above_ring_k": ["True", "False"]
}

# Streamlit App
st.title("Mushroom Classification App 🍄")
st.write("Predict whether a mushroom is **poisonous** or **edible** based on its characteristics.")

st.header("Enter Mushroom Characteristics")

# Create selectboxes for the top features
user_inputs = {}
for feature in top_features:
    user_inputs[feature] = st.selectbox(
        f"{feature}",
        options=["Select a value"] + unique_values.get(feature, ["Unknown"])
    )

# Predict Button
if st.button("Predict"):
    # Check if all values are selected
    if "Select a value" in user_inputs.values():
        st.error("Please select all values before making a prediction.")
    else:
        # Convert user inputs into the format needed for the model
        input_data = []
        for feature in feature_importance_df['Feature']:
            if feature in user_inputs:
                input_data.append(1 if user_inputs[feature] == "True" else 0)
            else:
                input_data.append(0)  # Default to 0 for missing features

        # Make prediction
        prediction = model.predict([input_data])[0]
        prediction_proba = model.predict_proba([input_data])[:, 1][0]

        # Display the result
        if prediction == 1:
            st.error(f"The mushroom is **poisonous**! (Probability: {prediction_proba:.2f})")
        else:
            st.success(f"The mushroom is **edible**! (Probability: {1 - prediction_proba:.2f})")