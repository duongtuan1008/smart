# train_classifier.py
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

with open("embeddings.pkl", "rb") as f:
    X, y = pickle.load(f)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000)
clf.fit(X_train, y_train)

print("✅ Huấn luyện hoàn tất.")
print("🎯 Accuracy:", accuracy_score(y_test, clf.predict(X_test)))

with open("clf.pkl", "wb") as f:
    pickle.dump(clf, f)
