import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# Excel file read karna
df = pd.read_excel(
    r"C:\Users\ROHIT\Downloads\house_price_dataset(1).xlsx"
)

print(df.head())

# Dataset information
print(df.shape)
print(df.columns)
print(df.dtypes)

# Missing values check karna
print(df.isnull().sum())
print(df[df.isnull().any(axis=1)])


# Missing values fill karna
df["area"] = df["area"].fillna(df["area"].mean())
df["age"] = df["age"].fillna(df["age"].mean())

print(df.isnull().sum())


# Features aur Target
X = df[["area", "bedrooms", "age"]]
y = df["price"]

print(X)
print(y)


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Model banana
model = LinearRegression()


# Model ko train karna
model.fit(X_train, y_train)


# New house ki prediction
new_house = pd.DataFrame({
    "area": [1600],
    "bedrooms": [3],
    "age": [4]
})

prediction = model.predict(new_house)

print("Predicted House Price:", prediction[0], "lakh")


# Coefficients aur Intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


# Test data par prediction
predictions = model.predict(X_test)


# Actual vs Predicted
result = pd.DataFrame({
    "Actual Price": y_test,
    "Predicted Price": predictions
})

print(result)


# Model Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("R2 Score:", r2)