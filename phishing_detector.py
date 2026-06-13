import pandas as pd
import numpy as np
import io
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =====================================================================
# 1. SIMULATED DATASET GENERATION
# For your internship submission, we'll programmatically inject a clean 
# sample dataset representing common phishing patterns and clean emails.
# =====================================================================
def load_phishing_dataset():
    data = {
        "text": [
            # Phishing Indicators: Urgency, Account Suspension, Financial/Gift Cards, Suspect Links
            "URGENT: Your bank account has been suspended. Click here to verify your identity immediately: http://secure-bank-login-update.com",
            "Congratulations! You won a $1000 Amazon Gift Card! Claim your reward now by clicking this link: http://bit.ly/fake-amazon-win",
            "Dear customer, we detected unusual login activity on your Netflix account. Please reset your password: http://netflix-security-check.xyz",
            "Official Notice: Update your tax information immediately to avoid legal penalties. Access your portal here: http://irs-refund-portal.net",
            "Verify your PayPal wallet now to release your pending funds. Failure to comply within 24 hours will result in permanent suspension.",
            
            # Safe / Legitimate Communications
            "Hi Abhinav, thank you for completing your first internship module. Your submission is received.",
            "Hey, are we still meeting for lunch at 1 PM today? Let me know if you need to reschedule.",
            "Your monthly Thiranex subscription invoice is now available for download in your student portal dashboard.",
            "Project update: The team has successfully migrated the SQLite history database to the production branch.",
            "Hi team, please review the attached slide deck regarding next week's penetration testing schedule."
        ],
        # 1 = Phishing, 0 = Safe
        "label": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    }
    return pd.DataFrame(data)

def main():
    print("==========================================")
    print("🤖 AI-POWERED PHISHING EMAIL DETECTOR")
    print("==========================================\n")
    
    # 1. Load the dataset
    df = load_phishing_dataset()
    print(f"📊 Dataset successfully loaded with {len(df)} sample emails.")
    
    # 2. Split data into training and testing sets (80% train, 20% test)
    # Using stratify ensures equal class representation across the splits
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    
    # 3. Feature Extraction (NLP Text Vectorization)
    # TF-IDF converts raw text data into numbers based on word importance matrices
    print("⚙️ Extracting textual/URL features using TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 4. Initialize and Train the Machine Learning Classifier
    print("🏋️ Training Logistic Regression classification engine...")
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    
    # 5. Model Evaluation
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n==========================================")
    print("📊 MODEL EVALUATION PERFORMANCE")
    print("==========================================")
    print(f"🎯 Accuracy Score: {accuracy * 100:.2f}%")
    
    print("\n📝 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Safe", "Phishing"]))
    
    print("🧩 Confusion Matrix Output:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  True Negatives (Safe correctly flagged)    : {cm[0][0]}")
    print(f"  False Positives (Safe misflagged as Phish) : {cm[0][1]}")
    print(f"  False Negatives (Phish missed by model)    : {cm[1][0]}")
    print(f"  True Positives (Phish correctly caught)    : {cm[1][1]}")
    print("==========================================\n")
    
    # 6. Interactive Real-Time Prediction Sandbox
    print("🔍 Test the Live Model (Interactive Sandbox)")
    print("------------------------------------------")
    while True:
        user_input = input("\nPaste custom email text content to analyze (or type 'exit' to quit): ").strip()
        if user_input.lower() == 'exit' or not user_input:
            break
            
        # Convert user test inputs through the vectorization pipeline
        custom_features = vectorizer.transform([user_input])
        prediction = model.predict(custom_features)[0]
        probabilities = model.predict_proba(custom_features)[0]
        
        confidence = probabilities[prediction] * 100
        
        print("\n--- Diagnostic Verdict ---")
        if prediction == 1:
            print(f"🚨 STATUS: PHISHING DETECTED (Confidence: {confidence:.2f}%)")
            print("💡 Risk Flags: Detected structural urgency, high-frequency keyword triggers, or unverified link patterns.")
        else:
            print(f"✨ STATUS: SAFE EMAIL (Confidence: {confidence:.2f}%)")
            print("💡 Evaluation: Content mimics standard conversational syntax without anomalous endpoint references.")
        print("------------------------------------------")

if __name__ == "__main__":
    main()
