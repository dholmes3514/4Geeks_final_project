'''Web app for price and turnaround time prediction'''

import pickle
import streamlit as st
from datetime import datetime, timedelta
import plotly.io as pio
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Price & Turnaround Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Plotly defaults for dark theme and lime accent
# Fetch the plotly_dark template defensively and assign to a custom key if available.
try:
    try:
        dark_tmpl = pio.templates["plotly_dark"]
    except Exception:
        # some environments expose templates differently; fallback to default attr
        dark_tmpl = getattr(pio.templates, "default", None)

    if dark_tmpl is not None:
        pio.templates["custom_dark"] = dark_tmpl
except Exception:
    # if anything fails here, continue without registering custom template
    pass

px.defaults.template = "plotly_dark"
px.defaults.color_continuous_scale = [[0, 'lightgray'], [1, '#9be15d']]

# Inject custom CSS and Google Fonts for a sleeker look (navy + lime accents)
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
    /* Page background and font tweaks */
    .stApp {
        background: linear-gradient(180deg, #07123b 0%, #0b234a 100%);
        color: #fff;
        font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    /* Hide the Streamlit footer and menu */
    footer {visibility: hidden;} 
    #MainMenu {visibility: hidden;}
    /* Sidebar styling (wider, semi-transparent) */
    [data-testid="stSidebar"] > div:first-child { background-color: rgba(255,255,255,0.03) !important; padding: 18px; }
    [data-testid="stSidebar"] .css-1v0mbdj { padding-top: 1.5rem; }
    [data-testid="stSidebar"] .stButton>button { border-radius: 10px; }
    /* Hero header */
    .hero-title { font-size: 34px; font-weight: 800; margin: 0; color: #ffffff; }
    .hero-sub { color: #cbd5e1; margin-top:4px; font-size:12px }
    /* Section cards */
    .section-card { background: rgba(255,255,255,0.02); border-radius: 12px; padding: 16px; box-shadow: 0 6px 18px rgba(2,6,23,0.5); margin-bottom:16px }
    /* Metric value accent */
    .stMetricValue, .metric-value { color: #e6ffb3 !important; font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True,
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

            # Use model.predict (helper functions return the first value already)
            predicted_price = predict_price(preprocessed_data, pp)
            predicted_turnaround_time = predict_tt(preprocessed_data)

            # Ensure numeric types for display
            predicted_price = float(predicted_price)
            predicted_turnaround_time = float(predicted_turnaround_time)
            
            def format_turnaround_days(days: float) -> str:
                """Convert fractional days into a human readable string and ETA.

                Returns a short string like '57 min (ETA 15:32)' or '1d 3h (ETA 12:05)'.
                """
                total_minutes = int(round(days * 24 * 60))
                if total_minutes < 60:
                    human = f"{total_minutes} min"
                elif total_minutes < 24 * 60:
                    hours = total_minutes // 60
                    minutes = total_minutes % 60
                    human = f"{hours}h {minutes}m" if minutes else f"{hours}h"
                else:
                    days_i = total_minutes // (24 * 60)
                    rem = total_minutes % (24 * 60)
                    hours = rem // 60
                    human = f"{days_i}d {hours}h" if hours else f"{days_i}d"

                # Compute ETA based on local time (format: MM/DD/YYYY hh:mm AM/PM)
                try:
                    eta = datetime.now() + timedelta(minutes=total_minutes)
                    eta_str = eta.strftime("%m/%d/%Y %I:%M %p")
                    return f"{human} (ETA {eta_str})"
                except Exception:
                    return human

            # Display results in main panel
            st.subheader("📈 Prediction Results")
            col1, col2 = st.columns(2)
            col1.metric("💰 Estimated Price", f"${predicted_price:,.2f}")
            # Display formatted turnaround time (model returns days)
            col2.metric("⏱️ Turnaround Time", format_turnaround_days(predicted_turnaround_time))
            st.write(f"Raw turnaround value (days): {predicted_turnaround_time}")

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

                fig = px.choropleth(
                    all_states,
                    locations='state_code',
                    locationmode='USA-states',
                    color='value',
                    scope='usa',
                    color_continuous_scale=[[0, 'lightgray'], [1, 'crimson']],
                    range_color=(0, 1),
                    labels={'value': ''},
                )
                fig.update_layout(coloraxis_showscale=False, margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info('State not recognized for mapping. Try a full state name like "Kentucky".')
        except Exception as e:
            st.error(f"Prediction failed: {e}")
