# Badminton Reviews Classification

 This project performs sentiment analysis on badminton product reviews. It includes a machine learning pipeline to classify reviews into **Positive**, **Neutral**, or **Negative** sentiments and a Streamlit web application for real-time inference.

## 📂 Project Structure

- **`data_cleaning.ipynb`**: Jupyter notebook for initial data cleaning and exploration.
- **`train_and_save.py`**: Python script to preprocess data, train the Random Forest model, and save the pipeline.
- **`deploy_folder/`**: Contains the application code and artifacts for deployment.
    - **`app.py`**: Streamlit application for the user interface.
    - **`flask_app/`**: (Optional) Flask version of the app.
    - **`sentiment_pipeline.joblib`**: The trained model pipeline.
    - **`requirements.txt`**: Python dependencies.
- **`steps_to_deploy_to_aws`**: Guide for deploying the application to AWS EC2.

## 🧠 Model Details

The sentiment analysis model is built using **Scikit-learn**.
- **Preprocessing**: Text cleaning, stopword removal, and lemmatization using NLTK.
- **Feature Extraction**: Bag-of-Words (CountVectorizer).
- **Classifier**: Random Forest Classifier.
- **Classes**:
    - 0: Negative
    - 1: Neutral
    - 2: Positive

## 🚀 Installation

1. Clone the repository (if applicable) or navigate to the project directory.
2. Navigate to the deployment folder:
   ```bash
   cd deploy_folder
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

To run the Streamlit web application locally:

```bash
streamlit run app.py
```

The app will launch in your default web browser at `http://localhost:8501`.

## ☁️ Deployment

For deployment instructions on AWS EC2, please refer to the `steps_to_deploy_to_aws` file in the root directory. It covers:
- EC2 instance creation.
- Security group configuration.
- Copying files to the server.
- Setting up the environment and running the app.
