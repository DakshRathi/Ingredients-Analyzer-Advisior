# app.py
import os
import streamlit as st
import requests
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Health Advisor",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Styling ---
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stSpinner > div > div {
        border-top-color: #4CAF50;
    }
    h1, h2, h3 {
        color: #2E8B57; /* SeaGreen */
    }
</style>
""", unsafe_allow_html=True)

# --- UI Layout ---
st.title("🍎 AI Health Advisor")
st.markdown(
    "Upload a picture of a food's ingredient list, and our AI will provide a comprehensive health analysis. "
    "Get insights into benefits, potential risks, and healthier alternatives in seconds."
)
st.divider()

# --- Backend API Configuration ---
# Make sure your FastAPI server is running at this address.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/analyze/")

# Initialize session state to hold the analysis result
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# --- File Uploader and Analysis Trigger ---
uploaded_file = st.file_uploader(
    "Choose an image of an ingredient list...",
    type=["png", "jpg", "jpeg"],
    help="For best results, use a clear, well-lit photo of the flat ingredient panel."
)

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns([1, 2])
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        if st.button("🔍 Analyze Ingredients", use_container_width=True):
            st.session_state.analysis_result = None # Clear previous results
            with st.spinner("Our AI agents are analyzing the ingredients... This may take a moment."):
                try:
                    # Prepare the file for the POST request
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(API_URL, files=files, timeout=90) # 90-second timeout

                    if response.status_code == 200:
                        st.session_state.analysis_result = response.json()
                    else:
                        st.error(f"Error from server ({response.status_code}): {response.text}")
                        st.session_state.analysis_result = None
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to connect to the analysis server: {e}")
                    st.session_state.analysis_result = None


# --- Display Analysis Results ---
if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    st.divider()
    st.header("🔬 Analysis Report")

    extracted_data = result.get("extracted_data", {})
    
    # Handle cases where the image is not a valid food ingredient list
    if extracted_data.get("validation_status") != "valid_food_image":
        st.warning(f"**Validation Failed:** {extracted_data.get('error_message', 'Could not analyze the image.')}")
        st.info("Please upload a clear image of a food product's ingredient list.")
    else:
        # --- Main Summary ---
        product_name = extracted_data.get("product_name") or "The Product"
        st.subheader(f"Summary for: {product_name}")
        st.markdown(f"_{result.get('final_summary_message_for_user', 'Here is your analysis.')}_")

        # --- Nutritional Info ---
        nutritional_info = extracted_data.get("nutritional_info")
        if nutritional_info:
            st.subheader("Key Nutritional Facts (per 100g)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Calories", f"{nutritional_info.get('calories_per_100g', 'N/A')} kcal")
            c2.metric("Protein", f"{nutritional_info.get('protein_grams', 'N/A')} g")
            c3.metric("Carbs", f"{nutritional_info.get('carbohydrates_grams', 'N/A')} g")
            c4.metric("Fat", f"{nutritional_info.get('fat_grams', 'N/A')} g")

        # --- Tabbed Layout for Detailed Analysis ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "✅ Benefits", "⚠️ Disadvantages", "🩺 Disease Associations", 
            "🔄 Healthier Alternatives", "📋 Extracted Data"
        ])

        with tab1:
            benefits = result.get("benefits_analysis", {})
            st.write(f"**Confidence:** {benefits.get('confidence_level', 'N/A')}")
            for finding in benefits.get('findings', []):
                st.success(finding)
            with st.expander("See Detailed Analysis"):
                st.write(benefits.get('detailed_analysis', 'No detailed analysis provided.'))

        with tab2:
            disadvantages = result.get("disadvantages_analysis", {})
            st.write(f"**Confidence:** {disadvantages.get('confidence_level', 'N/A')}")
            for finding in disadvantages.get('findings', []):
                st.error(finding)
            with st.expander("See Detailed Analysis"):
                st.write(disadvantages.get('detailed_analysis', 'No detailed analysis provided.'))

        with tab3:
            disease = result.get("disease_analysis")
            if disease:
                st.write(f"**Confidence:** {disease.get('confidence_level', 'N/A')}")
                for finding in disease.get('findings', []):
                    st.warning(finding)
                with st.expander("See Detailed Analysis"):
                    st.write(disease.get('detailed_analysis', 'No detailed analysis provided.'))
            else:
                st.info("No disease associations found for this product.")

        with tab4:
            alternatives = result.get("alternatives_report", {})
            st.info(alternatives.get("summary", "No summary for alternatives provided."))
            alt_list = alternatives.get("alternatives", [])
            for alt in alt_list:
                st.markdown(f"**{alt.get('product_name', 'Unknown Alternative')}**")
                st.write(f"_{alt.get('reason', 'No reason provided.')}_")

        with tab5:
            st.write("**Product Brand:**", extracted_data.get("brand", "Not detected"))
            st.write("**Detected Ingredients:**")
            st.info(", ".join(extracted_data.get("ingredients", ["None"])))
            st.write("**Detected Allergens:**")
            st.warning(", ".join(extracted_data.get("allergens", ["None"])))
