import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Dataset where both features matter
data = {
    "study_hours": [2, 4, 6, 8, 10],
    "attendance": [50, 60, 70, 80, 90],
    "marks": [30, 50, 70, 85, 100]
}

df = pd.DataFrame(data)

# Features and target
X = df[["study_hours", "attendance"]]
y = df["marks"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
pickle.dump(model, open("model.pkl", "wb"))
print("Model trained with both features affecting marks")