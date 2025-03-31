# Drug Concentration Overlap Analysis

A Streamlit application for analyzing and visualizing pharmacokinetic (PK) concentration-time profiles and calculating overlap scores between different compounds.

## Features

- **Data Input**: Upload CSV files containing PK parameters or use sample data
- **Concentration-Time Curve Visualization**: Plot and compare concentration-time curves
- **Overlap Score Calculation**: Calculate and visualize the overlap between different compounds
- **Image Processing**: Extract concentration-time data from uploaded plot images
- **Formula Reference**: Built-in PK formula documentation
- **Multiple Administration Routes**: Support for both IV and oral administration
- **Normalization**: Option to view normalized concentrations for better comparison

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/drug-concentration-overlap.git
cd drug-concentration-overlap
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
streamlit run src/app.py
```

2. Use the application:
   - Upload your CSV data file or use the sample data
   - Select compounds to compare
   - Adjust visualization settings
   - View overlap scores and concentration-time curves

### CSV Data Format

Your CSV file should include the following columns:
- `No.`: Compound identifier
- `Name`: Compound name
- `Dose[mg]`: Dose amount (can include /kg)
- `Admin route`: Administration route (IV or Oral)
- `Cmin[mg/mL]`: Minimum concentration
- `Cmax[mg/mL]`: Maximum concentration
- `Tmax[h]`: Time to maximum concentration
- `Thalf[h]`: Half-life

### Pharmacokinetic Models

The application uses:
- First-order elimination model for IV administration
- Two-compartment model with first-order absorption for oral administration
- Area-based overlap score calculation for comparing concentration profiles

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.