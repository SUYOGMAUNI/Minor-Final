import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc,precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # Import SMOTE for oversampling
from algo import RandomForest  # Import your custom RandomForest algorithm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("diabetes.csv")

df.info()
print(f"Number of Rows {df.shape[0]} and number of columns {df.shape[1]}")

# Function to calculate median for each target group
def cal_median(df, target, var):
    temp = df[df[var] != 0]  # Only consider non-zero values for median calculation
    return temp.groupby(target)[var].median()

# Function for median imputation
def median_imputation(df, target, var):
    """
    Replace zero values in the specified column (var) with the median of non-zero values
    for each target group in the dataset.
    """
    medians = cal_median(df, target, var)  # Get median values

    df.loc[(df[target] == 0) & (df[var] == 0), var] = medians[0]
    df.loc[(df[target] == 1) & (df[var] == 0), var] = medians[1]

# Apply median imputation for relevant columns
columns_to_impute = ["Glucose", "SkinThickness", "BloodPressure", "BMI", "Insulin"]

for col in columns_to_impute:
    median_imputation(df, "Outcome", col)

# **Verify that zero values are removed**
print("\nChecking for remaining zero values after median imputation:")
print((df[columns_to_impute] == 0).sum())

# Save updated dataset
df.to_csv("updated_diabetes.csv", index=False)
print("\nUpdated dataset saved as 'updated_diabetes.csv'")

# Prepare the unscaled dataset
y = df["Outcome"].values
X = df.drop("Outcome", axis=1)  # Features without scaling

# Split the data (Still unscaled)
X_train_unscaled, X_test_unscaled, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to the unscaled dataset BEFORE SCALING
smote = SMOTE(sampling_strategy={0: 500, 1: 500}, random_state=42)  # Specify 500 samples for both classes
X_train_resampled_unscaled, y_train_resampled = smote.fit_resample(X_train_unscaled, y_train)

# Convert the resampled dataset to a DataFrame
df_resampled_unscaled = pd.DataFrame(X_train_resampled_unscaled, columns=X.columns)
df_resampled_unscaled["Outcome"] = y_train_resampled  # Add target column

# Convert specific columns to whole numbers (integers)
integer_columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "Age"]
df_resampled_unscaled[integer_columns] = df_resampled_unscaled[integer_columns].round().astype(int)

# **Apply rounding to 'BMI' and 'DiabetesPedigreeFunction'**
df_resampled_unscaled['BMI'] = df_resampled_unscaled['BMI'].round(1)  # Round BMI to 1 decimal place
df_resampled_unscaled['DiabetesPedigreeFunction'] = df_resampled_unscaled['DiabetesPedigreeFunction'].round(3)  # Round to 3 decimal places

# **Verify again that no zero values exist**
print("\nChecking for zero values after SMOTE:")
print((df_resampled_unscaled[columns_to_impute] == 0).sum())

# Save the unscaled resampled dataset to CSV
output_path_unscaled = r"C:\Users\Suyog\Desktop\checkpre\resampled_diabetes_unscaled.csv"
df_resampled_unscaled.to_csv(output_path_unscaled, index=False)

# Display the first few rows of the unscaled dataset
print(df_resampled_unscaled.tail())

# Check class distribution
print("Class distribution after SMOTE (Unscaled):")
print(df_resampled_unscaled["Outcome"].value_counts())

print(f"Unscaled resampled dataset saved at: {output_path_unscaled}")

# Now apply scaling AFTER saving the unscaled dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled_unscaled)  # Apply scaling to resampled training data
X_test_scaled = scaler.transform(X_test_unscaled)  # Apply scaling to the original test data

# Check the data used for training and testing
print(f"Original Training Data Size (before SMOTE): {X_train_unscaled.shape[0]}")
print(f"Resampled Training Data Size (after SMOTE): {X_train_resampled_unscaled.shape[0]}")
print(f"Test Data Size: {X_test_unscaled.shape[0]}")

# Initialize the custom Random Forest model
model = RandomForest(n_trees=30, max_depth=7, min_samples_split=15, min_samples_leaf=15)

# Train the model on the resampled and scaled data
model.fit(X_train_scaled, y_train_resampled)

# Save the trained model and scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Evaluate performance on training data
y_train_pred = model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train_resampled, y_train_pred)

# Evaluate performance on test data
y_test_pred = model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Calculate additional metrics
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

# Print results
print(f"Training Accuracy: {train_accuracy * 100:.4f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.4f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Check for overfitting
if train_accuracy > test_accuracy + 0.10:  # 10% gap threshold
    print("Potential Overfitting Detected: Model performs much better on training data.")
elif test_accuracy > train_accuracy:
    print("Underfitting Detected: Model generalizes poorly to both training and test data.")
else:
    print("Model is generalizing well.")

# --- ADDITIONAL METRICS ---
import matplotlib.pyplot as plt
import seaborn as sns

# # Class distribution before SMOTE
# plt.figure(figsize=(6, 4))
# sns.countplot(x='Outcome', data=df, palette={'0': 'green', '1': 'red'})
# plt.title('Class Distribution Before SMOTE')
# plt.xlabel('Outcome (0: No Diabetes, 1: Diabetes)')
# plt.ylabel('Count')
# plt.show()
#
# # Class distribution after SMOTE
# plt.figure(figsize=(6, 4))
# sns.countplot(x=y_train_resampled, palette={'0': 'green', '1': 'red'})
# plt.title('Class Distribution After SMOTE')
# plt.xlabel('Outcome (0: No Diabetes, 1: Diabetes)')
# plt.ylabel('Count')
# plt.show()

# Pairwise Correlation Heatmap
# Define the feature columns you want to use for correlation matrix
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Calculate the correlation matrix
correlation_matrix = df[features].corr()

# Plot the heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title('Pairwise Correlation Heatmap')
plt.show()


# Compute ROC curve and AUC
# fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)
# roc_auc = auc(fpr, tpr)
#
# # Plot the ROC curve
# plt.figure(figsize=(6, 5))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc='lower right')
# plt.show()
# Define features (excluding the target variable 'Outcome')
# Define features (excluding the target variable 'Outcome')
# Plotting feature distributions for key features
# features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']
#
# df[features].hist(bins=20, figsize=(14, 10))
# plt.suptitle('Feature Distributions')
# plt.show()
# Calculate precision, recall, and thresholds
# precision, recall, thresholds = precision_recall_curve(y_test, y_test_pred)
#
# # Plot the precision-recall curve
# plt.figure(figsize=(6, 5))
# plt.plot(recall, precision, marker='.')
# plt.title('Precision-Recall Curve')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.show()
# Store the accuracies over each step

#Initialize lists to store accuracy values
epochs = 10  # Set number of epochs to train for
train_accuracy_list = []
test_accuracy_list = []

# Repeat the training for multiple epochs
for epoch in range(1, epochs + 1):
    print(f"Epoch {epoch}/{epochs}")

    # Train the model on the resampled and scaled data
    model.fit(X_train_scaled, y_train_resampled)

    # Evaluate performance on training data
    y_train_pred = model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train_resampled, y_train_pred)

    # Evaluate performance on test data
    y_test_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Store accuracy values for plotting
    train_accuracy_list.append(train_accuracy)
    test_accuracy_list.append(test_accuracy)

    # Optionally print accuracy for this epoch
    print(f"Training Accuracy: {train_accuracy * 100:.4f}%")
    print(f"Testing Accuracy: {test_accuracy * 100:.4f}%")
#
# Plotting the accuracy graph
plt.plot(range(1, epochs + 1), train_accuracy_list, label='Training Accuracy', marker='o', color='blue')
plt.plot(range(1, epochs + 1), test_accuracy_list, label='Testing Accuracy', marker='o', color='green')

plt.title('Model Accuracy Over Multiple Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
#
# Show plot
plt.show()

# # Apply PCA for dimensionality reduction (optional)
# pca = PCA(n_components=5)  # You can experiment with different numbers of components
# X_train_scaled_pca = pca.fit_transform(X_train_scaled)
# X_test_scaled_pca = pca.transform(X_test_scaled)
#
# # Train your model using the transformed PCA features
# model.fit(X_train_scaled_pca, y_train_resampled)
# # Make predictions on the test data using the PCA-transformed features
# y_test_pred_pca = model.predict(X_test_scaled_pca)
#
# # Evaluate the model's performance
# test_accuracy_pca = accuracy_score(y_test, y_test_pred_pca)
# precision_pca = precision_score(y_test, y_test_pred_pca)
# recall_pca = recall_score(y_test, y_test_pred_pca)
# f1_pca = f1_score(y_test, y_test_pred_pca)
#
# # Print evaluation metrics
# print(f"Test Accuracy (PCA): {test_accuracy_pca * 100:.4f}%")
# print(f"Precision (PCA): {precision_pca:.4f}")
# print(f"Recall (PCA): {recall_pca:.4f}")
# print(f"F1 Score (PCA): {f1_pca:.4f}")
#
# # Generate confusion matrix
# from sklearn.metrics import confusion_matrix
# cm_pca = confusion_matrix(y_test, y_test_pred_pca)
# sns.heatmap(cm_pca, annot=True, fmt="d", cmap="Blues", xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
# plt.title("Confusion Matrix (PCA)")
# plt.show()
#
# # Optionally: Classification report
# from sklearn.metrics import classification_report
# print("Classification Report (PCA):")
# print(classification_report(y_test, y_test_pred_pca))

