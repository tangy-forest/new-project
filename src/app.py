import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import cv2
from PIL import Image
import io
from scipy.signal import savgol_filter

# Set page config
st.set_page_config(
    page_title="Concentration Overlap",
    page_icon="💊",
    layout="wide"
)

def process_dose(dose_str):
    """
    Process dose string to extract numerical value and unit
    Returns: (float or None, bool) - (dose value, is_per_kg)
    """
    if pd.isna(dose_str) or dose_str == '':
        return None, False
        
    # Convert to string if number
    dose_str = str(dose_str).strip()
    
    # Check if dose is per kg
    is_per_kg = '/kg' in dose_str.lower()
    
    try:
        # Remove /kg and convert to float
        numeric_str = dose_str.lower().replace('/kg', '').strip()
        return float(numeric_str), is_per_kg
    except ValueError:
        st.warning(f"Could not parse dose value: {dose_str}")
        return None, False

def test_dose_processing():
    """
    Test the dose processing function with various inputs
    """
    test_cases = [
        ('10/kg', (10.0, True)),
        ('10', (10.0, False)),
        (10, (10.0, False)),
        ('140.27', (140.27, False)),
        ('9.90/kg', (9.90, True)),
        ('', (None, False)),
        (np.nan, (None, False)),
        ('30000', (30000.0, False))
    ]
    
    results = []
    for input_val, expected in test_cases:
        result = process_dose(input_val)
        passed = result == expected
        results.append({
            'input': str(input_val) if not pd.isna(input_val) else 'NaN',
            'expected': f"{expected[0]}/kg" if expected[0] is not None and expected[1] else str(expected[0]),
            'got': f"{result[0]}/kg" if result[0] is not None and result[1] else str(result[0]),
            'passed': passed,
            'expected_raw': expected,
            'got_raw': result
        })
    
    df = pd.DataFrame(results)
    
    # Add explanation column for failed tests
    df['explanation'] = df.apply(lambda row: 
        '' if row['passed'] else 
        f"Expected {row['expected_raw']}, got {row['got_raw']}", axis=1)
    
    # Return only the display columns
    return df[['input', 'expected', 'got', 'passed', 'explanation']]

def preprocess_data(df):
    """
    Preprocess the dataframe:
    1. Forward fill missing doses for same compounds
    2. Remove rows with missing essential data
    3. Handle NA/missing values in Tmax for IV routes
    4. Process dose values
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Clean up column names by removing spaces
    df.columns = df.columns.str.replace(' ', '')
    
    # Process doses and add per_kg indicator
    dose_info = df['Dose[mg]'].apply(process_dose)
    df['dose_value'] = [x[0] for x in dose_info]
    df['is_per_kg'] = [x[1] for x in dose_info]
    
    # Forward fill doses within same compound
    df['dose_value'] = df.groupby('Name')['dose_value'].fillna(method='ffill')
    df['is_per_kg'] = df.groupby('Name')['is_per_kg'].fillna(method='ffill')
    
    # Convert 'NA' strings to np.nan
    df = df.replace('NA', np.nan)
    
    # Set Tmax to 0 for IV routes
    df.loc[df['Adminroute'] == 'IV', 'Tmax[h]'] = 0
    
    # Remove rows with missing essential data
    essential_columns = ['Cmin[mg/mL]', 'Cmax[mg/mL]', 'Thalf[h]']
    df = df.dropna(subset=essential_columns)
    
    return df

def create_sample_data():
    """Create a sample DataFrame with the given format"""
    data = {
        'No.': [1, 1, 2, 4, 4, 6, 6, 7],
        'Name': ['Dantrolene', 'Dantrolene', 'Amlodipine', 'Amlodipine', 'Quinine', 'Quinine', '18β-glycyrrhetinic acid', '18β-glycyrrhetinic acid'],
        'Dose[mg]': [10, 10, 15, 5, 1200, np.nan, 140.27, np.nan],
        'Admin route': ['Oral', 'IV', 'Oral', 'IV', 'Oral', 'IV', 'Oral', 'IV'],
        'Cmin[mg/mL]': [0.362, 0.293, 0.00492, 0.00091, 0.779, 0.860, 0.110, 0.155],
        'Cmax[mg/mL]': [4.866, 42.120, 0.04533, 0.039, 4.567, 6.814, 0.110, 0.222],
        'Tmax[h]': [3.580, np.nan, 3.15, np.nan, 2.360, np.nan, 23.500, np.nan],
        'Thalf[h]': [3.430, 3.270, 4.42, 4.36, 7.960, 7.960, 43.570, 48.140]
    }
    return pd.DataFrame(data)

def calculate_concentration(t, params, route="oral"):
    """
    Calculate concentration at time t given PK parameters
    
    params: dict containing Cmax, Cmin, Tmax, Thalf for oral
                        or C0 (Cmax), Thalf for IV
    route: "oral" or "iv"
    t: numpy array of time points
    """
    if route.lower() == "iv":
        k = np.log(2) / params["Thalf"]
        return params["Cmax"] * np.exp(-k * t)
    else:
        # Use a more realistic absorption model with sigmoidal uptake
        k_elim = np.log(2) / params["Thalf"]
        tmax = params["Tmax"]
        
        # Calculate ka (absorption rate constant) based on Tmax and k_elim
        # Using the relationship: ka = ln(2)/Tmax * (1 + sqrt(1 + 2*Tmax*k_elim/ln(2)))
        ka = np.log(2)/tmax * (1 + np.sqrt(1 + 2*tmax*k_elim/np.log(2)))
        
        # Two-compartment model with first-order absorption
        F = 1.0  # Bioavailability
        conc = params["Cmax"] * (
            (ka/(ka - k_elim)) * 
            (np.exp(-k_elim * t) - np.exp(-ka * t))
        )
        
        return conc

def calculate_overlap_score(conc1, conc2, time_points):
    """
    Calculate the overlap score between two concentration curves
    Returns percentage overlap
    """
    # Normalize concentrations to 0-1 range
    norm1 = conc1 / np.max(conc1)
    norm2 = conc2 / np.max(conc2)
    
    # Calculate overlap area
    min_curve = np.minimum(norm1, norm2)
    max_curve = np.maximum(norm1, norm2)
    
    overlap_area = np.trapz(min_curve, time_points)
    total_area = np.trapz(max_curve, time_points)
    
    return (overlap_area / total_area) * 100

def plot_concentration_curves(data, reference_name="Dantrolene", show_normalized=True):
    """
    Plot concentration curves for reference drug vs selected compounds
    """
    fig = go.Figure()
    
    # Time points for plotting (24 hours with 1000 points)
    time_points = np.linspace(0, 24, 1000)
    
    # Get reference drug data (both IV and Oral if available)
    ref_data = data[data['Name'] == reference_name]
    ref_concs = []
    
    for _, row in ref_data.iterrows():
        params = {
            "Cmin": row['Cmin[mg/mL]'],
            "Cmax": row['Cmax[mg/mL]'],
            "Tmax": 0 if row['Adminroute'].lower() == 'iv' else float(row['Tmax[h]']),
            "Thalf": float(row['Thalf[h]'])
        }
        
        conc = calculate_concentration(time_points, params, row['Adminroute'])
        ref_concs.append(conc)
        
        if show_normalized:
            conc = conc / np.max(conc)
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=conc,
            name=f"{row['Name']} ({row['Adminroute']}, {row['Dose[mg]']}mg)",
            line=dict(color='purple', width=2),
            fill='tozeroy',
            fillcolor='rgba(128, 0, 128, 0.2)'
        ))
    
    # Update layout to match the reference plot
    fig.update_layout(
        title=f"Plasma Concentration - {reference_name}",
        xaxis=dict(
            title="Time [h]",
            gridcolor='lightgray',
            showgrid=True,
            zeroline=True,
            zerolinecolor='lightgray'
        ),
        yaxis=dict(
            title="Concentration [mg/mL]" if not show_normalized else "Normalized Concentration",
            gridcolor='lightgray',
            showgrid=True,
            zeroline=True,
            zerolinecolor='lightgray'
        ),
        plot_bgcolor='white',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    return fig, time_points, ref_concs

def detect_axis_ranges(image):
    """
    Automatically detect axis ranges from the plot image
    Returns: tuple of (x_min, x_max, y_min, y_max)
    """
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale if image is RGB
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Get image dimensions
    height, width = gray.shape
    
    # Apply thresholding to isolate text and lines
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Find grid lines using Hough transform
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    if lines is None:
        return 0, 24, 0, 5000  # Default values if no lines detected
    
    # Find the leftmost vertical line (y-axis) and bottom horizontal line (x-axis)
    x_axis_y = height - 50  # Approximate y-coordinate of x-axis
    y_axis_x = 50          # Approximate x-coordinate of y-axis
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < 10:  # Horizontal line
            if y1 > height * 0.8:  # Near bottom of image
                x_axis_y = min(y1, x_axis_y)
        if abs(x2 - x1) < 10:  # Vertical line
            if x1 < width * 0.2:  # Near left of image
                y_axis_x = max(x1, y_axis_x)
    
    # Add padding for axis detection
    x_padding = int(width * 0.05)
    y_padding = int(height * 0.05)
    
    # Adjust coordinates to account for padding
    plot_width = width - y_axis_x - x_padding
    plot_height = x_axis_y - y_padding
    
    return 0, 24, 0, 5000  # For now return standard ranges, will implement OCR in next iteration

def extract_curve_from_image(image, x_range=None, y_range=None):
    """
    Extract concentration-time curve data from an uploaded plot image with improved accuracy
    
    Args:
        image: PIL Image object
        x_range: tuple of (min, max) for x-axis, if None will auto-detect
        y_range: tuple of (min, max) for y-axis, if None will auto-detect
    Returns:
        tuple of (time_points, concentrations)
    """
    # Auto-detect axis ranges if not provided
    if x_range is None or y_range is None:
        x_min, x_max, y_min, y_max = detect_axis_ranges(image)
        x_range = x_range or (x_min, x_max)
        y_range = y_range or (y_min, y_max)
    
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale if image is RGB
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Get image dimensions
    height, width = gray.shape
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to handle varying brightness
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,  # Block size
        2    # C constant
    )
    
    # Remove grid lines
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    # Filter contours by size and shape
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if area > 100 and perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.8:
                valid_contours.append(contour)
    
    if not valid_contours:
        raise ValueError("No valid curve found in the image")
    
    # Find the longest valid contour that spans a reasonable x-range
    curve_points = None
    max_x_span = 0
    
    for contour in valid_contours:
        points = contour.squeeze()
        if len(points.shape) < 2:
            continue
            
        x_coords = points[:, 0]
        x_span = np.ptp(x_coords)
        
        # The curve should span at least 50% of the image width
        if x_span > width * 0.5 and x_span > max_x_span:
            curve_points = points
            max_x_span = x_span
    
    if curve_points is None:
        raise ValueError("No suitable curve found spanning the time axis")
    
    # Sort points by x coordinate
    curve_points = curve_points[curve_points[:, 0].argsort()]
    
    # Remove duplicate x-coordinates by averaging y-values
    unique_x = np.unique(curve_points[:, 0])
    averaged_points = np.array([
        (x, np.mean(curve_points[curve_points[:, 0] == x, 1]))
        for x in unique_x
    ])
    
    # Add padding to avoid edge effects
    x_padding = 0.05 * width
    y_padding = 0.05 * height
    
    # Convert coordinates with padding compensation
    time_points = (averaged_points[:, 0] - x_padding) / (width - 2 * x_padding) * (x_range[1] - x_range[0]) + x_range[0]
    concentrations = ((height - averaged_points[:, 1] - y_padding) / (height - 2 * y_padding)) * (y_range[1] - y_range[0]) + y_range[0]
    
    # Ensure curve starts at time 0
    # Find the closest point to time 0
    start_idx = np.argmin(np.abs(time_points))
    time_points = time_points[start_idx:]
    concentrations = concentrations[start_idx:]
    
    # Interpolate to ensure we have a point exactly at t=0
    if time_points[0] > 0:
        time_points = np.concatenate([[0], time_points])
        concentrations = np.concatenate([[0], concentrations])
    
    # Apply Savitzky-Golay filter to smooth the curve
    window = min(51, len(time_points) - 1 if len(time_points) % 2 == 0 else len(time_points) - 2)
    if window > 3:
        concentrations = savgol_filter(concentrations, window, 3)
    
    # Ensure values are within the specified ranges
    time_points = np.clip(time_points, x_range[0], x_range[1])
    concentrations = np.clip(concentrations, y_range[0], y_range[1])
    
    return time_points, concentrations

def extract_plot_info(image):
    """
    Extract plot information (title, axis labels, legend) using OCR
    Returns: dict with plot metadata
    """
    # For now, return default values
    # In a full implementation, we would use OCR to extract this information
    return {
        'title': 'Extracted Plot',
        'x_label': 'Time [h]',
        'y_label': 'Concentration [ng/mL]',
        'compound': 'Unknown'
    }

def plot_extracted_curve(time_points, concentrations, plot_info):
    """
    Create an interactive plot of the extracted curve data
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time_points,
        y=concentrations,
        mode='lines',
        name='Extracted curve',
        line=dict(color='blue', width=2)
    ))
    
    # Add markers for key points
    max_idx = np.argmax(concentrations)
    min_idx = np.argmin(concentrations)
    
    fig.add_trace(go.Scatter(
        x=[time_points[max_idx]],
        y=[concentrations[max_idx]],
        mode='markers+text',
        name='Cmax',
        text=[f'Cmax: {concentrations[max_idx]:.1f}'],
        textposition='top right',
        marker=dict(size=10, color='red'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=[time_points[min_idx]],
        y=[concentrations[min_idx]],
        mode='markers+text',
        name='Cmin',
        text=[f'Cmin: {concentrations[min_idx]:.1f}'],
        textposition='bottom right',
        marker=dict(size=10, color='green'),
        showlegend=False
    ))
    
    fig.update_layout(
        title=plot_info['title'],
        xaxis=dict(
            title=plot_info['x_label'],
            gridcolor='lightgray',
            showgrid=True,
            zeroline=True,
            zerolinecolor='lightgray'
        ),
        yaxis=dict(
            title=plot_info['y_label'],
            gridcolor='lightgray',
            showgrid=True,
            zeroline=True,
            zerolinecolor='lightgray'
        ),
        plot_bgcolor='white',
        hovermode='x unified'
    )
    
    return fig

def main():
    st.title("Drug Concentration Overlap Analysis")
    
    # Add formula section
    with st.expander("📚 Pharmacokinetic Formulas", expanded=False):
        st.markdown("""
        ## IV Administration
        Concentration after intravenous (IV) administration follows first-order elimination:

        $$
        C(t) = C_0 \cdot e^{-k_e t}
        $$

        where:
        * $C(t)$ is the concentration at time $t$
        * $C_0$ is the initial concentration
        * $k_e = \\frac{\\ln(2)}{t_{1/2}}$ is the elimination rate constant
        * $t_{1/2}$ is the half-life

        ## Oral Administration
        Concentration after oral administration follows a two-compartment model with first-order absorption:

        $$
        C(t) = C_{max} \cdot \\frac{k_a}{k_a - k_e} \cdot (e^{-k_e t} - e^{-k_a t})
        $$

        where:
        * $k_a$ is the absorption rate constant
        * $k_a = \\frac{\\ln(2)}{t_{max}} \cdot \\left(1 + \sqrt{1 + \\frac{2t_{max}k_e}{\\ln(2)}}\\right)$
        * $t_{max}$ is the time to peak concentration

        ## Overlap Score
        The overlap between two concentration curves is calculated as:

        $$
        \\text{Overlap}(\\%) = \\frac{\\int_0^T \\min(C_1(t), C_2(t)) \\, dt}{\\int_0^T \\max(C_1(t), C_2(t)) \\, dt} \cdot 100
        $$

        where $C_1(t)$ and $C_2(t)$ are normalized concentration curves.
        """)
    
    # Create tabs
    tab1, tab2 = st.tabs(["Data Input", "Image Input"])
    
    # Data Input tab
    with tab1:
        # File upload for compound data
        st.subheader("Upload Compound Data")
        uploaded_file = st.file_uploader("Upload CSV with columns: No., Name, Dose[mg], Admin route, Cmin[mg/mL], Cmax[mg/mL], Tmax[h], Thalf[h]", type="csv")
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                # Clean up column names immediately after reading
                data.columns = data.columns.str.replace(' ', '')
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                return
        else:
            # Add sample data button
            if st.button("Load Sample Data"):
                data = create_sample_data()
            else:
                st.info("Please upload a CSV file with compound data or use the sample data.")
                return
        
        # Preprocess the data
        data = preprocess_data(data)
        
        # Display the data with dose information
        st.write("Compound Data:")
        display_data = data.copy()
        display_data['Processed Dose'] = display_data.apply(
            lambda x: f"{x['dose_value']}/kg" if x['is_per_kg'] else f"{x['dose_value']}", 
            axis=1
        )
        st.dataframe(display_data)
        
        # Visualization settings
        st.subheader("Visualization Settings")
        show_normalized = st.checkbox("Show normalized concentrations", value=True)
        admin_route_filter = st.multiselect(
            "Filter by administration route:",
            options=['All', 'Oral', 'IV'],
            default=['All']
        )
        
        # Filter data by admin route if needed
        if 'All' not in admin_route_filter:
            data = data[data['Adminroute'].isin(admin_route_filter)]
        
        # Select reference compound (default to Dantrolene)
        available_compounds = data['Name'].unique()
        default_index = list(available_compounds).index("Dantrolene") if "Dantrolene" in available_compounds else 0
        
        reference = st.selectbox(
            "Select reference compound:",
            available_compounds,
            index=default_index
        )
        
        # Create time points for plotting (24 hours with 1000 points)
        time_points = np.linspace(0, 24, 1000)
        
        # Create visualization
        fig = go.Figure()
        
        # Plot reference compound
        ref_data = data[data['Name'] == reference]
        ref_concs = []  # Initialize ref_concs list
        
        for _, row in ref_data.iterrows():
            params = {
                "Cmin": row['Cmin[mg/mL]'],
                "Cmax": row['Cmax[mg/mL]'],
                "Tmax": 0 if row['Adminroute'].lower() == 'iv' else float(row['Tmax[h]']),
                "Thalf": float(row['Thalf[h]'])
            }
            
            conc = calculate_concentration(time_points, params, row['Adminroute'])
            ref_concs.append(conc)  # Store reference concentrations
            
            if show_normalized:
                conc = conc / np.max(conc)
                
            fig.add_trace(go.Scatter(
                x=time_points,
                y=conc,
                name=f"{row['Name']} ({row['Adminroute']}, {row['Dose[mg]']}mg)",
                line=dict(color='purple', width=2),
                fill='tozeroy',
                fillcolor='rgba(128, 0, 128, 0.2)'
            ))
        
        # Update layout to match reference plot
        fig.update_layout(
            title=f"Plasma Concentration - {reference}",
            xaxis=dict(
                title="Time [h]",
                gridcolor='lightgray',
                showgrid=True,
                zeroline=True,
                zerolinecolor='lightgray'
            ),
            yaxis=dict(
                title="Concentration [mg/mL]" if not show_normalized else "Normalized Concentration",
                gridcolor='lightgray',
                showgrid=True,
                zeroline=True,
                zerolinecolor='lightgray'
            ),
            plot_bgcolor='white',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        
        # Add comparison compound
        comparison = st.selectbox("Select compound to compare:", 
                                [name for name in available_compounds if name != reference])
        
        if comparison:
            comp_data = data[data['Name'] == comparison]
            for i, row in enumerate(comp_data.iterrows()):
                row = row[1]  # Get the actual row data
                params = {
                    "Cmin": row['Cmin[mg/mL]'],
                    "Cmax": row['Cmax[mg/mL]'],
                    "Tmax": 0 if row['Adminroute'].lower() == 'iv' else float(row['Tmax[h]']),
                    "Thalf": float(row['Thalf[h]'])
                }
                
                conc = calculate_concentration(time_points, params, row['Adminroute'])
                if show_normalized:
                    conc = conc / np.max(conc)
                    
                fig.add_trace(go.Scatter(
                    x=time_points,
                    y=conc,
                    name=f"{row['Name']} ({row['Adminroute']}, {row['Dose[mg]']}mg)",
                    line=dict(color='gold', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255, 215, 0, 0.2)'
                ))
                
                # Calculate and display overlap score with corresponding reference curve
                if i < len(ref_concs):  # Make sure we have a reference curve to compare with
                    overlap = calculate_overlap_score(ref_concs[i], conc, time_points)
                    st.write(f"Overlap score between {reference} and {comparison} ({row['Adminroute']}): {overlap:.1f}%")
        
        st.plotly_chart(fig)

    # Image Input tab
    with tab2:
        st.write("Upload a concentration-time plot image")
        
        # Basic file uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["png", "jpg", "jpeg"]
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Plot")
            
            # Basic controls
            st.write("Extraction Settings")
            x_max = st.number_input("X-axis max (hours)", value=24.0)
            y_max = st.number_input("Y-axis max (concentration)", value=5000.0)
            
            if st.button("Extract Curve"):
                try:
                    # Extract and display curve
                    time_points, concentrations = extract_curve_from_image(
                        image,
                        x_range=(0, x_max),
                        y_range=(0, y_max)
                    )
                    
                    # Plot results
                    fig = plot_extracted_curve(
                        time_points, 
                        concentrations,
                        {'title': 'Extracted Plot', 'x_label': 'Time [h]', 'y_label': 'Concentration'}
                    )
                    st.plotly_chart(fig)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.info("Please upload an image file (PNG, JPG)")

if __name__ == "__main__":
    main() 