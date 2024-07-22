# Pipeline-with-Algorithmn
## Importing Libraries 
### from sklearn.compose import ColumnTransformer
### from sklearn.pipeline import Pipeline
### from sklearn.preprocessing import OneHotEncoder
### from sklearn.linear_model import LogisticRegression
### from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
### from sklearn.model_selection import train_test_split

# Assuming df is your DataFrame and 'Weather Type' is your target column
X = df.drop(columns=['Weather Type'])
y = df['Weather Type']

# Specify the categorical column names, excluding the target column
categorical_features = ['Cloud Cover', 'Season', 'Location']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 1: ColumnTransformer with OneHotEncoder
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse=False, drop='first'), categorical_features)
], remainder='passthrough')

# Step 2: Logistic Regression
step2 = LogisticRegression(max_iter=1000)  # Adding max_iter to ensure convergence

# Create the pipeline
pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

# Fit the pipeline on the training data
pipe.fit(X_train, y_train)

# Make predictions on the test data
y_pred = pipe.predict(X_test)

# Print classification metrics
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred, average='weighted'))
print('Recall:', recall_score(y_test, y_pred, average='weighted'))
print('F1 Score:', f1_score(y_test, y_pred, average='weighted'))
