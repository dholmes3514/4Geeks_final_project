'''Web app for price and turnaround time prediction'''

import pickle
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Price & Turnaround Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Flask stuff (wait for later)


# Load assets (Label encoder, one-hot encoder, price model, TT model)
with open('models/price_predictor.pkl', 'rb') as input_file:
    pp = pickle.load(input_file)

with open('models/turnaroundtime_predictor.pkl', 'rb') as input_file:
     tt = pickle.load(input_file)

data_df = pd.read_csv('data/training_data')

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


# 'Main fence'
if __name__ == '__main__':

    print('Running sales & TT app...\n')

    # Streamlit Dashboard Layout
    st.header("Input Project Details")
    st.sidebar.header("🔍 Input Parameters")
    ship_mode = st.sidebar.selectbox("Ship Mode", ["First Class", "Second Class", "Standard Class", "Same Day"])
    category = st.sidebar.selectbox("Category", ["Furniture", "Office Supplies", "Technology"])
    # Sub-category options depend on the selected category
    subcategory_map = {
        "Furniture": ["Bookcases", "Chairs", "Tables", "Furnishings", "Appliances", "Art"],
        "Office Supplies": ["Binders", "Paper", "Envelopes", "Fasteners", "Labels", "Storage", "Supplies", "Accessories"],
        "Technology": ["Phones", "Machines", "Copiers", "Accessories"]
    }
    sub_category_options = subcategory_map.get(category, [])
    # show a dependent selectbox; default to first option when available
    if sub_category_options:
        sub_category = st.sidebar.selectbox("Sub-Category", sub_category_options)
    else:
        sub_category = st.sidebar.text_input("Sub-Category")
    state = st.sidebar.text_input("State", "Kentucky")
    postal_code = st.sidebar.text_input("Postal Code", "42420")


    if st.sidebar.button("Run Prediction"):
        raw_input_df = pd.DataFrame([{
            "Ship_Mode": ship_mode,
            "State": state,
            "Category": category,
            "Sub_Category": sub_category,
            "Postal_Code": postal_code,
        }])
        try:
            preprocessed_data = preprocess_input(raw_input_df, encoder, labeler)
            # Debug: show preprocessed features and encoder feature names to validate inputs
            try:
                feature_names = encoder.get_feature_names_out(['Ship_Mode', 'Category', 'Sub_Category'])
            except Exception:
                feature_names = None
            st.debug = getattr(st, 'debug', None)
            st.write("--- Debug: preprocessed input features ---")
            st.dataframe(preprocessed_data)
            st.write("feature names from encoder:", feature_names)

            # Use model.predict and show raw arrays (helps diagnose zero predictions)
            predicted_price_array = pp.predict(preprocessed_data)
            predicted_tt_array = tt.predict(preprocessed_data)
            st.write("raw predicted_price array:", predicted_price_array)
            st.write("raw predicted_tt array:", predicted_tt_array)

            # Then take the first value for display
            predicted_price = float(predicted_price_array[0])
            predicted_turnaround_time = float(predicted_tt_array[0])
            # Display results in main panel
            st.subheader("📈 Prediction Results")
            col1, col2 = st.columns(2)
            col1.metric("💰 Estimated Price", f"${predicted_price:,.2f}")
            # Display model-returned turnaround time beside the price (raw model units assumed to be days)
            col2.metric("⏱️ Turnaround Time", f"{predicted_turnaround_time:.2f} days")
            st.write(f"Raw turnaround value: {predicted_turnaround_time}")
            st.write(raw_input_df.head())

            # --- Map: highlight selected state ---
            # Small mapping from common state names (lowercased) to USPS code
            state_to_abbrev = {
                'alabama': 'AL','alaska':'AK','arizona':'AZ','arkansas':'AR','california':'CA','colorado':'CO',
                'connecticut':'CT','delaware':'DE','florida':'FL','georgia':'GA','hawaii':'HI','idaho':'ID','illinois':'IL',
                'indiana':'IN','iowa':'IA','kansas':'KS','kentucky':'KY','louisiana':'LA','maine':'ME','maryland':'MD','massachusetts':'MA',
                'michigan':'MI','minnesota':'MN','mississippi':'MS','missouri':'MO','montana':'MT','nebraska':'NE','nevada':'NV',
                'new hampshire':'NH','new jersey':'NJ','new mexico':'NM','new york':'NY','north carolina':'NC','north dakota':'ND','ohio':'OH',
                'oklahoma':'OK','oregon':'OR','pennsylvania':'PA','rhode island':'RI','south carolina':'SC','south dakota':'SD','tennessee':'TN',
                'texas':'TX','utah':'UT','vermont':'VT','virginia':'VA','washington':'WA','west virginia':'WV','wisconsin':'WI','wyoming':'WY'
            }

            chosen = state.strip().lower()
            abbrev = state_to_abbrev.get(chosen, None)

            if abbrev is not None:
                # create a DataFrame with all states and highlight the selected one
                all_states = pd.DataFrame({
                    'state_code': list(state_to_abbrev.values()),
                    'value': [0]*len(state_to_abbrev)
                })
                all_states.loc[all_states['state_code'] == abbrev, 'value'] = 1

                fig = px.choropleth(all_states,
                                    locations='state_code',
                                    locationmode='USA-states',
                                    color='value',
                                    scope='usa',
                                    color_continuous_scale=[[0, 'lightgray'], [1, 'crimson']],
                                    range_color=(0,1),
                                    labels={'value':''})
                fig.update_layout(coloraxis_showscale=False, margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info('State not recognized for mapping. Try a full state name like "Kentucky".')
        except Exception as e:
            st.error(f"Prediction failed: {e}")
