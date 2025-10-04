import numpy as np
import pandas as pd
import pickle # To save the trained model

print("="*60)
print("Training Optimized Titanic Survival Model (5 features)")
print("="*60)

# Load data
df = pd.read_csv("Titanic_Dataset.csv")

# Keep only the BEST 5 features (removed SibSp & Parch)
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Embarked']]

print(f"\nDataset: {df.shape[0]} samples")
print(f"Features: Pclass, Sex, Age, Embarked")

# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encode categorical values
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Prepare data
X = df.drop('Survived', axis=1).values
y = df['Survived'].values.reshape(-1, 1)

# Save normalization parameters
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)

# Normalize
X = (X - X_mean) / X_std

# Train-test split (80-20)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize parameters
m, n = X_train.shape
weights = np.zeros((n, 1))
bias = 0
lr = 0.01
epochs = 2000

# Training
print(f"\nTraining with lr={lr}, epochs={epochs}")
print("Progress:")
for i in range(epochs):
    z = np.dot(X_train, weights) + bias
    y_pred = sigmoid(z)
    
    # Gradients
    dw = (1/m) * np.dot(X_train.T, (y_pred - y_train))
    db = (1/m) * np.sum(y_pred - y_train)
    
    # Update weights
    weights -= lr * dw
    bias -= lr * db

    if i % 500 == 0:
        print(f"  Epoch {i}/{epochs}")

# Evaluate on test set
y_pred_test = sigmoid(np.dot(X_test, weights) + bias)
y_pred_class = (y_pred_test > 0.5).astype(int)
test_accuracy = np.mean(y_pred_class == y_test)

print(f"\n{'='*60}")
print(f"✅ Training Complete!")
print(f"✅ Test Accuracy: {test_accuracy*100:.2f}%")
print(f"{'='*60}")

# Save model
model = {
    'weights': weights,
    'bias': bias,
    'X_mean': X_mean,
    'X_std': X_std,
    'feature_names': ['Pclass', 'Sex', 'Age', 'Embarked'],
    'accuracy': test_accuracy
}

with open('titanic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(f"\n✅ Model saved as 'titanic_model.pkl'")
print(f"   Features: {model['feature_names']}")
print(f"   Accuracy: {test_accuracy*100:.2f}%")