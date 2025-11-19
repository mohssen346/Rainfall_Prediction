# src/03_classification.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load clustered data
data = pd.read_csv("data/clustered_data.csv")

# Drop non-feature columns
data = data.drop(columns=["datetime", "rrr24"], errors="ignore")

# Train: years before 2017, Test: 2017
train_data = data[data['year'] < 2017]
test_data  = data[data['year'] == 2017]

X_train = train_data.drop('Cluster', axis=1)
y_train = train_data['Cluster']
X_test  = test_data.drop('Cluster', axis=1)
y_test  = test_data['Cluster']

# Models
models = {
    "Random Forest" : RandomForestClassifier(random_state=42, n_jobs=-1),
    "SVM"           : SVC(random_state=42),
    "KNN"           : KNeighborsClassifier(n_jobs=-1),
    "Naive Bayes"   : GaussianNB(),
    "MLP"           : MLPClassifier(random_state=42, max_iter=300, verbose=False)
}

print("=== Individual Model Performance ===\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} -> Accuracy: {acc:.4f}")

# Best ensemble: Voting Classifier (RF + MLP)
voting_clf = VotingClassifier(
    estimators=[
        ('rf', models['Random Forest']),
        ('mlp', models['MLP'])
    ],
    voting='soft'
)

voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)

print("\n=== Voting Classifier (Random Forest + MLP) ===")
print(f"Final Accuracy: {accuracy_score(y_test, y_pred_voting):.4f}")
print(classification_report(y_test, y_pred_voting))

# Save predictions
results_df = X_test.copy()
results_df['Actual_Cluster'] = y_test.values
results_df['Predicted_Cluster'] = y_pred_voting
results_df.to_csv("results/prediction_results_voting.csv", index=False)
print("\nPredictions saved to results/prediction_results_voting.csv")