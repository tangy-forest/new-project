import streamlit as st
import numpy as np
import plotly.graph_objects as go
import cv2
from PIL import Image
from scipy.signal import savgol_filter

st.set_page_config(
    page_title="PK Curve Extractor",
    page_icon="📈",
    layout="wide"
)

def extract_curve_from_image(image, x_range=(0, 24), y_range=(0, 5000)):
    """Extract curve data from image"""
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale if image is RGB
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Get dimensions
    height, width = gray.shape
    
    # Preprocess image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    # Find the longest contour
    curve_contour = max(contours, key=len)
    points = curve_contour.squeeze()
    
    # Sort points by x coordinate
    points = points[points[:, 0].argsort()]
    
    # Convert to data coordinates
    time_points = (points[:, 0] / width) * x_range[1]
    concentrations = ((height - points[:, 1]) / height) * y_range[1]
    
    # Smooth the curve
    window = min(51, len(time_points) - 1 if len(time_points) % 2 == 0 else len(time_points) - 2)
    if window > 3:
        concentrations = savgol_filter(concentrations, window, 3)
    
    return time_points, concentrations

def plot_curve(time_points, concentrations):
    """Create interactive plot"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time_points,
        y=concentrations,
        mode='lines',
        name='Extracted curve',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title="Extracted Concentration-Time Curve",
        xaxis_title="Time [h]",
        yaxis_title="Concentration",
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )
    
    return fig

def main():
    st.title("PK Curve Extractor")
    
    uploaded_file = st.file_uploader(
        "Upload a concentration-time plot image",
        type=["png", "jpg", "jpeg"]
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Plot", use_column_width=True)
        
        # Axis range inputs
        col1, col2 = st.columns(2)
        with col1:
            x_max = st.number_input("X-axis max (hours)", value=24.0, min_value=0.1)
        with col2:
            y_max = st.number_input("Y-axis max (concentration)", value=5000.0, min_value=0.1)
        
        if st.button("Extract Curve"):
            try:
                with st.spinner("Extracting curve..."):
                    # Extract curve data
                    time_points, concentrations = extract_curve_from_image(
                        image,
                        x_range=(0, x_max),
                        y_range=(0, y_max)
                    )
                    
                    # Plot results
                    fig = plot_curve(time_points, concentrations)
                    st.plotly_chart(fig)
                    
                    # Allow downloading data
                    data = np.column_stack([time_points, concentrations])
                    st.download_button(
                        "Download Data (CSV)",
                        data=str(data.tolist()),
                        file_name="extracted_curve.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error extracting curve: {str(e)}")
    else:
        st.info("Please upload an image file (PNG or JPG)")

if __name__ == "__main__":
    main() 