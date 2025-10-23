"""Web app for price and turnaround time prediction

This file runs a small Streamlit UI that loads pre-saved model artifacts from
`models/` and uses a canonical preprocessing function to build feature rows for
the predictors. The UI is intentionally simple so the app can be run locally
with `streamlit run src/app.py`.
"""

import pickle
import re
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Price & Turnaround Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Load artifacts ---------------------------------------------------------
models_dir = Path("models")

def load_pickle(path: Path, friendly: str = "artifact"):
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load {friendly} from {path}: {e}")
        raise

# required artifacts
pp = load_pickle(models_dir / "price_predictor.pkl", "price model")
tt = load_pickle(models_dir / "turnaroundtime_predictor.pkl", "turnaround-time model")

# optional sample data (used only for quick inspection)
data_df = None
try:
    data_df = pd.read_csv("data/training_data")
except Exception:
    # not fatal for the UI; keep going
    data_df = None

# label encoders (State, Postal_Code) and a OneHotEncoder saved as encoder.pkl
labeler = {}
for col in ("State", "Postal_Code"):
    labeler_path = models_dir / f"{col}_encoder.pkl"
    if labeler_path.exists():
        labeler[col] = load_pickle(labeler_path, f"{col} encoder")
    else:
        st.warning(f"Missing encoder: {labeler_path} — some features may fail at runtime")

encoder = None
enc_path = models_dir / "encoder.pkl"
if enc_path.exists():
    encoder = load_pickle(enc_path, "one-hot encoder")
else:
    st.warning("Missing encoder.pkl — categorical encoding will fail unless added to models/")


#############
# Helpers
#############

def preprocess_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s\-/]", "", text)
    return text.lower().strip()


def preprocess_input(df: pd.DataFrame, encoder, labeler) -> pd.DataFrame:
    """Return a DataFrame shaped for model.predict.

    - cleans text columns
    - applies label encoders for `State` and `Postal_Code` (expects 1-D arrays)
    - applies the saved OneHotEncoder to Ship_Mode/Category/Sub_Category
    """
    df = df.copy()

    cols_to_clean = ["Ship_Mode", "State", "Category", "Sub_Category"]
    for col in cols_to_clean:
        df[col] = df[col].astype(str).apply(preprocess_text)

    # Label encode State and Postal_Code using the preloaded labelers
    for col in ("State", "Postal_Code"):
        if col in labeler:
            # transform expects a 1-D array-like. Wrap result into a Series
            transformed = labeler[col].transform(df[col])
            # ensure Series aligned with df.index
            df[f"{col}_encoded"] = pd.Series(transformed, index=df.index)
        else:
            raise ValueError(f"Missing label encoder for {col}")

    # One-hot encode categorical triplet using preloaded encoder
    if encoder is None:
        raise ValueError("Missing OneHot encoder (encoder.pkl) — cannot encode categorical features")

    cat_df = df[["Ship_Mode", "Category", "Sub_Category"]]
    encoded_array = encoder.transform(cat_df)
    # If encoder returns a sparse matrix, convert to dense
    if hasattr(encoded_array, "toarray"):
        encoded_array = encoded_array.toarray()

    # normalize to numpy array and ensure 2-D shape
    encoded_array = np.asarray(encoded_array)
    if encoded_array.ndim == 1:
        # single-row case: reshape to (1, n_features)
        if df.shape[0] == 1:
            encoded_array = encoded_array.reshape(1, -1)
        else:
            raise ValueError("OneHot encoder returned 1-D array for multiple rows")

    try:
        feature_names = encoder.get_feature_names_out(cat_df.columns)
    except Exception:
        # fallback for older sklearn versions
        feature_names = [f"enc_{i}" for i in range(encoded_array.shape[1])]

    encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)

    final_df = pd.concat([df[["State_encoded", "Postal_Code_encoded"]], encoded_df], axis=1)
    # defensive: if concat produced a Series (unexpected), coerce to single-row DataFrame
    if isinstance(final_df, pd.Series):
        final_df = final_df.to_frame().T
    return final_df


def format_turnaround(days_value: float) -> str:
    """Format a turnaround float (days) as human readable string.

    Example: 0.5 days -> "12.0 hours"
    """
    hours = days_value * 24
    if hours < 1:
        minutes = hours * 60
        return f"{minutes:.0f} min"
    if hours >= 24:
        return f"{days_value:.2f} days"
    return f"{hours:.1f} hours"


# --- Streamlit UI ----------------------------------------------------------
def main():
    st.title("Price and Turnaround Time Predictor")
    st.sidebar.header("Shipping Detail Inputs")

    ship_mode = st.sidebar.selectbox("Shipping Mode", ["First Class", "Second Class", "Standard Class", "Same Day"])
    category = st.sidebar.selectbox("Category", ["Furniture", "Office Supplies", "Technology"])

    subcategory_map = {
        "Furniture": ["Bookcases", "Chairs", "Tables", "Furnishings", "Appliances", "Art"],
        "Office Supplies": ["Binders", "Paper", "Envelopes", "Fasteners", "Labels", "Storage", "Supplies", "Accessories"],
        "Technology": ["Phones", "Machines", "Copiers", "Accessories"],
    }

    sub_category_options = subcategory_map.get(category, [])
    if sub_category_options:
        sub_category = st.sidebar.selectbox("Sub-Category", sub_category_options)
    else:
        sub_category = st.sidebar.text_input("Sub-Category")

    state = st.sidebar.text_input("State", "Kentucky")
    postal_code = st.sidebar.text_input("Postal Code", "42420")

    if st.sidebar.button("Run Prediction"):
        raw_input_df = pd.DataFrame([
            {
                "Ship_Mode": ship_mode,
                "State": state,
                "Category": category,
                "Sub_Category": sub_category,
                "Postal_Code": postal_code,
            }
        ])

        try:
            preprocessed = preprocess_input(raw_input_df, encoder, labeler)

            # Defensive checks: models expect 2D input. If a Series slipped through,
            # convert it to a single-row DataFrame.
            if isinstance(preprocessed, pd.Series):
                preprocessed = preprocessed.to_frame().T

            if not hasattr(preprocessed, "shape") or len(preprocessed.shape) != 2:
                raise ValueError(f"Preprocessed input must be 2D (n_samples, n_features); got type={type(preprocessed)} shape={getattr(preprocessed, 'shape', None)}")

            # If model exposes n_features_in_ we can give a clear error when shapes mismatch
            expected = None
            try:
                expected = getattr(pp, 'n_features_in_', None)
            except Exception:
                expected = None
            if expected is not None and preprocessed.shape[1] != expected:
                raise ValueError(f"Feature count mismatch for price model: expected {expected} features but got {preprocessed.shape[1]} columns. Check encoder and preprocessing.")

            # Predictions
            predicted_price_array = pp.predict(preprocessed)
            predicted_tt_array = tt.predict(preprocessed)

            predicted_price = float(predicted_price_array[0])
            predicted_turnaround_time = float(predicted_tt_array[0])

            # Display
            st.subheader("📈 Prediction Results")
            col1, col2 = st.columns(2)
            col1.metric("💰 Estimated Price", f"${predicted_price:,.2f}")
            col2.metric("⏱️ Turnaround Time", format_turnaround(predicted_turnaround_time))



            st.write(raw_input_df.head())

            # --- Map: highlight selected state ---
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


if __name__ == '__main__':
    main()
