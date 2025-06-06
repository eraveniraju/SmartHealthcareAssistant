# 🧠 Smart Healthcare Assistant

A Streamlit-based AI assistant that helps predict diseases, recommends possible treatments, and provides drug reviews — especially useful for people in underserved areas.

## 🚀 Features

- Disease prediction using machine learning
- Symptoms-based health guidance
- Drug reviews and medicine recommendations
- Clean and interactive Streamlit UI
- Lightweight and easy to deploy

## 📂 Project Structure

HealthcareAssistant/
│
│   ├── app.py                 # Main Streamlit app script

│   ├── models/                # Folder to store ML models (joblib, pickle files)
│   │    ├── model1.joblib
│   │    └── model2.joblib
│   ├── data/                  # Dataset files like CSVs used by the app
│   │    ├── drug_reviews.csv
│   │    └── medications.csv
├── requirements.txt 
└── README.md

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn / XGBoost
- Joblib (for saving models)

## ⚙️ How to Run

1. **Clone the repository**
    ```bash
    git clone https://github.com/eraveniraju/HealthcareAssistant.git
    cd HealthcareAssistant
    ```

2. **Install the dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Start the app**
    ```bash
    streamlit run app/app.py
    ```

## 📌 Use Case

This project provides a lightweight AI-powered tool that:
- Helps users identify potential health issues from symptoms
- Suggests relevant medications with reviews
- Supports proactive healthcare awareness, especially in areas lacking medical infrastructure

## 🧪 Sample Prediction Flow

1. User enters symptoms
2. App predicts the possible disease
3. Displays top medications and real user drug reviews
4. Suggests whether to consult a doctor (based on severity logic)

## 🧑‍💻 Author

**Raju Eraveni**  
📧 [Email](mailto:eraveaniraju@gmail.com)  
🔗 [LinkedIn](www.linkedin.com/in/eraveni-raju)

## 📄 License

This project is for educational and demo purposes. For production use, please consult medical professionals for clinical validations.

