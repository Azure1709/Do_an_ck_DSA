import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Đọc dữ liệu từ file CSV
data = pd.read_csv('data.csv')

# Giả sử cột cuối cùng là nhãn và các cột còn lại là đặc trưng
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Tạo và huấn luyện mô hình Decision Tree
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Dự đoán và tính độ chính xác của Decision Tree
y_pred_tree = decision_tree.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print(f"Độ chính xác của Decision Tree: {accuracy_tree:.2f}")

# Tạo và huấn luyện mô hình Random Forest
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

# Dự đoán và tính độ chính xác của Random Forest
y_pred_forest = random_forest.predict(X_test)
accuracy_forest = accuracy_score(y_test, y_pred_forest)
print(f"Độ chính xác của Random Forest: {accuracy_forest:.2f}")

# Sử dụng k-fold cross-validation để đánh giá chính xác hơn
scores_tree = cross_val_score(decision_tree, X, y, cv=10)
scores_forest = cross_val_score(random_forest, X, y, cv=10)

print(f"Độ chính xác trung bình của Decision Tree với cross-validation: {scores_tree.mean():.2f}")
print(f"Độ chính xác trung bình của Random Forest với cross-validation: {scores_forest.mean():.2f}")
