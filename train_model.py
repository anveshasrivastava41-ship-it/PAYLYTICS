import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle

# Load dataset
df = pd.read_csv("salaries.csv")

# Select useful columns
df = df[[
    "experience_level",
    "employment_type",
    "job_title",
    "company_location",
    "company_size",
    "salary_in_usd"
]]

# Encode categorical variables
job_encoder = LabelEncoder()
df["job_title"] = job_encoder.fit_transform(df["job_title"])

location_encoder = LabelEncoder()
df["company_location"] = location_encoder.fit_transform(df["company_location"])

employment_encoder = LabelEncoder()
df["employment_type"] = employment_encoder.fit_transform(df["employment_type"])

size_encoder = LabelEncoder()
df["company_size"] = size_encoder.fit_transform(df["company_size"])

experience_encoder = LabelEncoder()
df["experience_level"] = experience_encoder.fit_transform(df["experience_level"])


# Split features and target
X = df[[
    "experience_level",
    "employment_type",
    "job_title",
    "company_location",
    "company_size"
]]

y = df["salary_in_usd"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor()
model.fit(X_train, y_train)


# Train model
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

rf = RandomForestRegressor()

grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

model = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_) # model is optimized automaically

# Predict
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)

print("Model MAE:", mae)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")
pickle.dump(experience_encoder, open("experience_encoder.pkl", "wb"))
pickle.dump(job_encoder, open("job_encoder.pkl", "wb"))
pickle.dump(location_encoder, open("location_encoder.pkl", "wb"))
pickle.dump(employment_encoder, open("employment_encoder.pkl", "wb"))
pickle.dump(size_encoder, open("size_encoder.pkl", "wb"))


#This shows which factors affect salary the most
import matplotlib.pyplot as plt
import pandas as pd

# Feature importance
importances = model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print(importance_df)

# Plot
plt.figure(figsize=(8,5))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.title("Feature Importance in Salary Prediction")
plt.gca().invert_yaxis()
plt.show()
