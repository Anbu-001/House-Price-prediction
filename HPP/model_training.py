import pandas as pd
import numpy as np
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("Bengaluru_House_Data.csv")

# Drop unnecessary columns
df = df.drop(['area_type', 'society', 'balcony', 'availability'], axis=1)
df = df.dropna()

# Convert size (e.g., '2 BHK' or '4 Bedroom') to BHK
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))

# Convert total_sqft to numeric
def convert_sqft_to_num(x):
    try:
        if '-' in x:
            tokens = x.split('-')
            return (float(tokens[0]) + float(tokens[1])) / 2
        return float(x)
    except:
        return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
df = df.dropna()

# Drop the original 'size' column after extracting bhk
df = df.drop(['size'], axis=1)

# One-hot encode the 'location' column
dummies = pd.get_dummies(df['location'].str.lower(), drop_first=True)
df_model = pd.concat([df.drop('location', axis=1), dummies], axis=1)

# Split features and target
X = df_model.drop('price', axis=1)
y = df_model['price']

# Split data for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
with open('banglore_home_prices_model.pickle', 'wb') as f:
    pickle.dump(model, f)

# Save the data column names (for future prediction input formatting)
columns = {
    'data_columns': list(X.columns)
}
with open("columns.json", "w") as f:
    f.write(json.dumps(columns))

print("✅ Model and columns saved successfully!")


