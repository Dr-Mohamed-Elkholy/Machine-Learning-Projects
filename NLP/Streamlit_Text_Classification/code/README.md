# Text Classification Project

This project demonstrates a text classification pipeline using Python. It includes data preprocessing, model training, and prediction using various machine learning models (Logistic Regression, Random Forest, SVM, Naive Bayes). A Streamlit application is also included to provide a user interface for training models and making predictions.

## Project Structure

- **Input/**: Directory to store input files (e.g., Excel and CSV files).
- **Output/**: Directory to store output files (e.g., trained models, vectorizers).
- **config.py**: Configuration settings.
- **utils.py**: Utility functions for saving/loading files, preprocessing text, and vectorizing data.
- **main.py**: Script for training and predicting using different machine learning models.
- **streamlit_app.py**: Streamlit application for user interaction.
- **requirements.txt**: Required Python packages.
- **README.md**: Project documentation.

## Setup Instructions

1. **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd text-classification-project
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit application**:
    ```bash
    streamlit run streamlit_app.py
    ```

5. **Use the application**:
    - Go to the URL provided by Streamlit.
    - Choose the mode ("Train" or "Predict").
    - Upload an Excel or CSV file to train a new model or enter text to make predictions.
    - Select the desired machine learning model and specify the column names for text and labels.

## Notes

- Ensure that the input files are in the `Input` directory.
- The trained model and vectorizer will be saved in the `Output` directory.
