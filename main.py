import os
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import config
from utils import save_file, load_file, process_text, vectorize

# Function to get the model based on the name
def get_model(model_name):
    if model_name == "logistic_regression":
        return LogisticRegression()
    elif model_name == "random_forest":
        return RandomForestClassifier()
    elif model_name == "svm":
        return SVC(probability=True)
    elif model_name == "naive_bayes":
        return MultinomialNB()
    else:
        raise ValueError("Model not supported")

# Function to train the model
def train_model(X_train, X_test, y_train, y_test, model_name):
    model = get_model(model_name)
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_acc = round(accuracy_score(y_train, train_pred) * 100, 2)
    test_acc = round(accuracy_score(y_test, test_pred) * 100, 2)
    print(f"Train Accuracy: {train_acc}%")
    print(f"Test Accuracy: {test_acc}%")
    return model

# Function to train the model
def train(args):
    input_file = os.path.join(config.input_path, args.file_name)
    vect_file = os.path.join(config.output_path, f"{args.output_name}.pkl")
    model_file = os.path.join(config.output_path, f"{args.output_name}_{args.model_name}.pkl")

    # Read data
    if input_file.endswith('.xlsx'):
        data = pd.read_excel(input_file)
    else:
        data = pd.read_csv(input_file)
    
    data = data[[args.text_col, args.label_col]]
    reviews = list(data[args.text_col])
    reviews = [process_text(r, config.stem) for r in reviews]
    y = data[args.label_col]
    X_train, X_test, y_train, y_test, vectorizer = vectorize(reviews, y, vect=args.vectorizer, min_df=config.min_df, ng_low=config.ng_low, ng_high=config.ng_high, test_size=config.test_size, rs=config.rs)
    save_file(vect_file, vectorizer)
    model = train_model(X_train, X_test, y_train, y_test, args.model_name)
    save_file(model_file, model)

# Function to make predictions
def predict(args):
    vect_file = os.path.join(config.output_path, f"{args.model_name}.pkl")
    model_file = os.path.join(config.output_path, f"{args.model_name}_{args.model_name}.pkl")
    vect = load_file(vect_file)
    model = load_file(model_file)
    tokens = [process_text(args.text)]
    X = vect.transform(tokens)
    pred_prob = round(model.predict_proba(X)[0, 1] * 100, 2)
    sentiment = "positive" if pred_prob >= 50 else "negative"
    print(f"Text: {args.text}")
    print(f"Sentiment: {sentiment} ({pred_prob}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "predict"], required=True, help="Mode: train or predict")
    parser.add_argument("--file_name", type=str, help="Input file name for training")
    parser.add_argument("--vectorizer", type=str, default="bow", help="Vectorizer, one of - 'bow', 'bowb', 'ng', 'tf'")
    parser.add_argument("--output_name", type=str, default="model", help="Output file name for training")
    parser.add_argument("--text", type=str, help="Text for prediction")
    parser.add_argument("--model_name", type=str, choices=config.models, default="logistic_regression", help="Model name for prediction")
    parser.add_argument("--text_col", type=str, default=config.text_col, help="Name of the text column")
    parser.add_argument("--label_col", type=str, default=config.label_col, help="Name of the label column")
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "predict":
        predict(args)
