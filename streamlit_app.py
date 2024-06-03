import streamlit as st
import os
import pandas as pd
from utils import save_file, load_file, process_text, vectorize
from main import train_model, get_model
import config

def train_and_save_model(file, vectorizer_type, model_name, text_col, label_col):
    if file.name.endswith('.xlsx'):
        data = pd.read_excel(file)
    else:
        data = pd.read_csv(file)
    
    data = data[[text_col, label_col]]
    reviews = list(data[text_col])
    reviews = [process_text(r, config.stem) for r in reviews]
    y = data[label_col]
    X_train, X_test, y_train, y_test, vectorizer = vectorize(reviews, y, vect=vectorizer_type, min_df=config.min_df, ng_low=config.ng_low, ng_high=config.ng_high, test_size=config.test_size, rs=config.rs)
    save_file(os.path.join(config.output_path, "vectorizer.pkl"), vectorizer)
    model = train_model(X_train, X_test, y_train, y_test, model_name)
    save_file(os.path.join(config.output_path, f"model_{model_name}.pkl"), model)
    return model, vectorizer

def predict_text(text, model, vectorizer):
    tokens = [process_text(text)]
    X = vectorizer.transform(tokens)
    pred_prob = round(model.predict_proba(X)[0, 1] * 100, 2)
    sentiment = "positive" if pred_prob >= 50 else "negative"
    return sentiment, pred_prob

def main():
    st.title("M.Elkholy: Text Classification App")

    st.sidebar.title("Options")
    mode = st.sidebar.selectbox("Choose mode", ["Train", "Predict"])

    if mode == "Train":
        st.header("Train a new model")
        uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=["xlsx", "csv"])
        vectorizer_type = st.selectbox("Choose a vectorizer", ["bows", "bowb", "ng", "tf"])
        model_name = st.selectbox("Choose a model", config.models)
        text_col = st.text_input("Enter the name of the text column", config.text_col)
        label_col = st.text_input("Enter the name of the label column", config.label_col)

        if st.button("Train"):
            if uploaded_file is not None:
                model, vectorizer = train_and_save_model(uploaded_file, vectorizer_type, model_name, text_col, label_col)
                st.success("Model trained and saved successfully.")
            else:
                st.error("Please upload a file.")
    
    if mode == "Predict":
        st.header("Make a prediction")
        text = st.text_area("Enter text for prediction")
        model_name = st.selectbox("Choose a model", config.models)
        
        if st.button("Predict"):
            vectorizer_path = os.path.join(config.output_path, "vectorizer.pkl")
            model_path = os.path.join(config.output_path, f"model_{model_name}.pkl")
            
            if os.path.exists(vectorizer_path) and os.path.exists(model_path):
                vectorizer = load_file(vectorizer_path)
                model = load_file(model_path)
                sentiment, pred_prob = predict_text(text, model, vectorizer)
                st.write(f"Sentiment: {sentiment} ({pred_prob}%)")
            else:
                st.error("Model and vectorizer not found. Please train the model first.")

if __name__ == "__main__":
    main()
