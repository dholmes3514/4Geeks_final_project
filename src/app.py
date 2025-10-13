'''Web app for price and turnaround time prediction'''

import pickle
import streamlit as st

st.title("🎯 Price & Turnaround Time Predictor")
st.markdown("Use this tool to estimate project timelines and costs based on work type and label.")

# Flask stuff (wait for later)


# Load assets (Label encoder, one-hot encoder, price model, TT model)
with open('models/price_predictor.pkl', 'rb') as input_file:
    pp = pickle.load(input_file)

with open('models/turnaroundtime_predictor.pkl', 'rb') as input_file:
     tt = pickle.load(input_file)

# Load encoders

labeler = {}
for col in ['State', 'Postal_Code']:
    with open(f'models/{col}_encoder.pkl', 'rb') as input_file:
        labeler[col] = pickle.load(input_file)
                                
with open('models/encoder.pkl', 'rb') as input_files:
    encoder = pickle.load(input_files)


#############
# Functions #
#############

import pandas as pd
import re

def preprocess_text(text):
    text = str(text).strip()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s\-/]", "", text)
    return text.lower().strip()

def preprocess_input(df, encoder, labeler):
    df = df.copy()

    cols_to_clean = ["Ship_Mode", "State", "Category", "Sub_Category"]

    for col in cols_to_clean:
        df[col] = df[col].astype(str).apply(preprocess_text)

    # Label Encoding using preloaded labeler 
    for col in labeler:
        df[f'{col}_encoded'] = labeler[col].transform(df[col].to_frame())

    # One-Hot Encoding using preloaded encoder
    cat_df = df[['Ship_Mode', 'Category', 'Sub_Category']]
    encoded_array = encoder.transform(cat_df)
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(cat_df.columns), index=df.index)

    # Combine all features
    final_df = pd.concat([df[['State_encoded', 'Postal_Code_encoded']], encoded_df], axis=1)

    return final_df


def predict_price(input_data, model):
    '''Takes preprocessed input data, runs price prediction, returns predicted price'''
    result = model.predict(input_data)[0]
    return result


def predict_tt(input_data):
    '''Takes preprocessed input data, runs price prediction, returns predicted turnaround time'''
    result = tt.predict(input_data)[0]
    return result

# Streamlit Dashboard Layout
st.header("Input Project Details")

col1, col2 = st.columns(2)
with col1:
    ship_mode = st.selectbox("Ship Mode", ["First Class", "Second Class", "Standard Class", "Same Day"])
    category = st.selectbox("Category", ["Furniture", "Office Supplies", "Technology"])
    sub_category = st.selectbox("Sub-Category", ["Bookcases", "Chairs", "Phones", "Binders", "Tables", "Storage", "Supplies", "Machines", "Copiers", "Accessories", "Furnishings", "Paper", "Appliances", "Art", "Envelopes", "Fasteners", "Labels"])
with col2:
    state = st.text_input("State", "Kentucky")
    postal_code = st.text_input("Postal Code", "42420")

if st.button("Predict Price & Turnaround Time"):
    raw_input_df = pd.DataFrame([{
        "Ship_Mode": ship_mode,
        "State": state,
        "Category": category,
        "Sub_Category": sub_category,
        "Postal_Code": postal_code,
    }])
    try:
        preprocessed_data = preprocess_input(raw_input_df, encoder, labeler)
        predicted_price = predict_price(preprocessed_data, pp)
        predicted_turnaround_time = predict_tt(preprocessed_data)
        st.success(f"Predicted Price: ${predicted_price:,.2f}")
        st.info(f"Predicted Turnaround Time: {predicted_turnaround_time} days")
    except Exception as e:
        st.error(f"Prediction failed: {e}")