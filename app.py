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
    .block-container {
        padding-top: 2rem;
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
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/analyze/")

# Initialize session state
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# --- Helper function for formatting the summary ---
def format_summary(result: dict) -> str:
    """Formats the final summary message with proper line breaks for Streamlit."""
    summary_parts = []

    benefits_analysis = result.get("benefits_analysis", {})
    if benefits_analysis and benefits_analysis.get('findings'):
        # Take the first one or two findings for the summary
        summary_parts.append(f"**✅ Benefits:** {'. '.join(benefits_analysis['findings'][:2])}")
    
    disadvantages_analysis = result.get("disadvantages_analysis", {})
    if disadvantages_analysis and disadvantages_analysis.get('findings'):
        summary_parts.append(f"**⚠️ Concerns:** {'. '.join(disadvantages_analysis['findings'][:2])}")

    alternatives_report = result.get("alternatives_report", {})
    if alternatives_report and alternatives_report.get('alternatives'):
        # Suggest the first alternative
        first_alt = alternatives_report['alternatives'][0].get('product_name', 'healthier options')
        summary_parts.append(f"**🔄 Alternatives:** Consider options like {first_alt}.")
    elif alternatives_report:
         summary_parts.append(f"**🔄 Alternatives:** {alternatives_report.get('summary', '')}")

    # Join the parts with markdown-compatible line breaks ("  \n")
    # This ensures each part is on a new line.
    return "  \n".join(summary_parts) if summary_parts else "Analysis complete. See tabs for details."


# --- File Uploader and Analysis Trigger ---
uploaded_file = st.file_uploader(
    "Choose an image of an ingredient list...",
    type=["png", "jpg", "jpeg"],
    help="For best results, use a clear, well-lit photo of the flat ingredient panel."
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        if st.button("🔍 Analyze Ingredients", use_container_width=True):
            st.session_state.analysis_result = None
            with st.spinner("Our AI agents are analyzing the ingredients... This may take a moment."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(API_URL, files=files, timeout=120) # Increased timeout for complex analyses
                    if response.status_code == 200:
                        st.session_state.analysis_result = response.json()
                    else:
                        st.error(f"Error from server ({response.status_code}): {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to connect to the analysis server: {e}")

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
        product_name = extracted_data.get("product_name") or "The Product"
        st.subheader(f"Analysis Summary for: {product_name}")
        
        # Use the new helper function to display the summary
        formatted_summary = format_summary(result)
        st.markdown(formatted_summary)

        st.subheader("Detailed Breakdown")
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "✅ Benefits", "⚠️ Disadvantages", "🩺 Disease Associations", 
            "🔄 Healthier Alternatives", "📋 Extracted Data"
        ])

        with tab1:
            benefits = result.get("benefits_analysis", {})
            if benefits and benefits.get('findings'):
                st.write(f"**Confidence:** {benefits.get('confidence_level', 'N/A')}")
                for finding in benefits.get('findings', []):
                    st.success(finding)
                with st.expander("See Detailed Explanation"):
                    st.markdown(benefits.get('detailed_analysis', 'No detailed analysis provided.'))
            else:
                st.info("No specific benefits were identified or an error occurred during analysis.")

        with tab2:
            disadvantages = result.get("disadvantages_analysis", {})
            if disadvantages and disadvantages.get('findings'):
                st.write(f"**Confidence:** {disadvantages.get('confidence_level', 'N/A')}")
                for finding in disadvantages.get('findings', []):
                    st.error(finding)
                with st.expander("See Detailed Explanation"):
                    st.markdown(disadvantages.get('detailed_analysis', 'No detailed analysis provided.'))
            else:
                st.info("No significant disadvantages were identified or an error occurred during analysis.")
        
        with tab3:
            disease = result.get("disease_analysis")
            if disease and disease.get('findings'):
                st.write(f"**Confidence:** {disease.get('confidence_level', 'N/A')}")
                for finding in disease.get('findings', []):
                    st.warning(finding)
                with st.expander("See Detailed Explanation"):
                    st.markdown(disease.get('detailed_analysis', 'No detailed analysis provided.'))
            else:
                st.info("No specific disease associations were found for this product.")

        with tab4:
            alternatives = result.get("alternatives_report", {})
            st.info(alternatives.get("summary", "No summary for alternatives provided."))
            alt_list = alternatives.get("alternatives", [])
            if alt_list:
                for alt in alt_list:
                    st.markdown(f"**{alt.get('product_name', 'Unknown Alternative')}**")
                    st.write(f"_{alt.get('reason', 'No reason provided.')}_")
                    st.markdown("---")
            else:
                st.success("No specific alternatives were recommended, which may indicate the product is reasonably healthy.")

        with tab5:
            st.write("**Product Brand:**", extracted_data.get("brand", "Not detected"))
            st.write("**Detected Ingredients:**")
            st.info(", ".join(extracted_data.get("ingredients", ["None"])))
            st.write("**Detected Allergens:**")
            st.warning(", ".join(extracted_data.get("allergens", ["None"])))
            # Display nutritional info again for completeness
            nutritional_info = extracted_data.get("nutritional_info")
            if nutritional_info:
                st.write("**Nutritional Info (per 100g):**")
                st.json(nutritional_info)
