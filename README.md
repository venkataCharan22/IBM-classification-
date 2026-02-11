# Customer Churn Analysis Dashboard

A comprehensive Streamlit dashboard for analyzing and predicting customer churn using machine learning models.

## Features

- **Overview**: Project introduction and dataset summary
- **EDA (Exploratory Data Analysis)**: Interactive visualizations and statistical analysis
- **Preprocessing Lab**: Data preprocessing and feature engineering tools
- **Model Arena**: Compare multiple machine learning models
- **Cross Validation**: Evaluate model performance with cross-validation
- **Overfitting Lab**: Analyze and prevent model overfitting

## Installation

1. Clone the repository:
```bash
git clone https://github.com/venkataCharan22/IBM-classification-.git
cd IBM-classification-
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

## Project Structure

```
├── app.py                      # Main application file
├── requirements.txt            # Python dependencies
├── Telco-Customer-Churn.csv   # Dataset
├── utils/                      # Utility modules
│   ├── data_loader.py         # Data loading functions
│   ├── models.py              # ML model implementations
│   ├── plotting.py            # Visualization functions
│   └── preprocessing.py       # Data preprocessing functions
└── views/                      # Dashboard views
    ├── overview.py            # Overview page
    ├── eda.py                 # EDA page
    ├── preprocessing_lab.py   # Preprocessing page
    ├── model_arena.py         # Model comparison page
    ├── cross_validation.py    # Cross-validation page
    └── overfitting_lab.py     # Overfitting analysis page
```

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- Plotly
- Seaborn
- Matplotlib

## License

This project is open source and available under the MIT License.
