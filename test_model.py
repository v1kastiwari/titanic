import numpy as np
import pandas as pd
import pickle

print("="*60)
print("Titanic Survival Predictor - Loading Model")
print("="*60)

# Load the trained model
with open('titanic_model.pkl', 'rb') as f:
    model = pickle.load(f)

weights = model['weights']
bias = model['bias']
X_mean = model['X_mean']
X_std = model['X_std']
feature_names = model['feature_names']
saved_accuracy = model['accuracy']

print(f"\n✅ Model loaded successfully!")
print(f"   Features: {', '.join(feature_names)}")
print(f"   Trained accuracy: {saved_accuracy*100:.2f}%")

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_passenger(pclass, sex, age, embarked):
    """
    Predict survival for a passenger
    
    Parameters:
    - pclass: 1, 2, or 3
    - sex: 'male' or 'female'
    - age: age in years
    - embarked: 'S' (Southampton), 'C' (Cherbourg), or 'Q' (Queenstown)
    """
    # Encode inputs
    sex_encoded = 0 if sex.lower() == 'male' else 1
    embarked_map = {'S': 0, 'C': 1, 'Q': 2}
    embarked_encoded = embarked_map.get(embarked.upper(), 0)
    
    # Create feature array (only 4 features now!)
    features = np.array([[pclass, sex_encoded, age, embarked_encoded]])
    
    # Normalize using saved parameters
    features_norm = (features - X_mean) / X_std
    
    # Predict
    prob = sigmoid(np.dot(features_norm, weights) + bias)[0][0]
    prediction = "SURVIVED" if prob > 0.5 else "DIED"
    
    return prediction, prob

# Verify on test dataset
print(f"\n{'='*60}")
print("Verifying on Test Dataset")
print("="*60)

df = pd.read_csv("Titanic_Dataset.csv")
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Embarked']]

# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encode
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

X = df.drop('Survived', axis=1).values
y = df['Survived'].values.reshape(-1, 1)

# Normalize
X = (X - X_mean) / X_std

# Test on same split
split = int(0.8 * len(X))
X_test = X[split:]
y_test = y[split:]

# Predict
y_pred_test = sigmoid(np.dot(X_test, weights) + bias)
y_pred_class = (y_pred_test > 0.5).astype(int)

accuracy = np.mean(y_pred_class == y_test)
print(f"\n✅ Verified Accuracy: {accuracy * 100:.2f}%")

# Sample Predictions
print(f"\n{'='*60}")
print("Sample Predictions")
print("="*60)

# Example 1: 3rd class male
print("\n1. Third class, Male, 22 years old, Southampton")
print("   (Similar to Jack from Titanic movie)")
pred, prob = predict_passenger(3, 'male', 22, 'S')
print(f"   → {pred} (probability: {prob*100:.1f}%)")

# Example 2: 1st class female
print("\n2. First class, Female, 30 years old, Cherbourg")
print("   (Similar to Rose from Titanic movie)")
pred, prob = predict_passenger(1, 'female', 30, 'C')
print(f"   → {pred} (probability: {prob*100:.1f}%)")

# Example 3: 2nd class male
print("\n3. Second class, Male, 35 years old, Southampton")
pred, prob = predict_passenger(2, 'male', 35, 'S')
print(f"   → {pred} (probability: {prob*100:.1f}%)")

# Example 4: 3rd class female with children
print("\n4. Third class, Female, 25 years old, Queenstown")
pred, prob = predict_passenger(3, 'female', 25, 'Q')
print(f"   → {pred} (probability: {prob*100:.1f}%)")

# Interactive prediction
print(f"\n{'='*60}")
print("Make Your Own Prediction")
print("="*60)

try:
    print("\nEnter passenger details:")
    pclass = int(input("  Passenger class (1, 2, or 3): "))
    sex = input("  Sex (male/female): ")
    age = float(input("  Age: "))
    embarked = input("  Port of embarkation (S/C/Q): ")
    
    pred, prob = predict_passenger(pclass, sex, age, embarked)
    
    print(f"\n{'─'*60}")
    print(f"🚢 Prediction: {pred}")
    print(f"   Survival probability: {prob*100:.1f}%")
    print(f"   Death probability: {(1-prob)*100:.1f}%")
    print(f"{'─'*60}")
    
except KeyboardInterrupt:
    print("\n\nExiting...")
except:
    print("\nInvalid input or skipping interactive mode...")

print(f"\n{'='*60}")
print("Analysis complete!")
print("="*60)