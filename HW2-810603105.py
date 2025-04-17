
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from scipy.stats import zscore
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import math

file_path = "Housing.csv"  # assuming it's in same folder
df = pd.read_csv(file_path)

print("📊 اطلاعات اولیه:")
df.info()
print("\n🔍 پنج سطر اول:")
print(df.head(5))
df.head(5).to_excel("first_5_rows.xlsx", index=False)

print("\n🧩 مقادیر گمشده به تفکیک ستون:")
print(df.isnull().sum().sort_values(ascending=False))

missing_percent = df.isnull().mean()
columns_to_drop = missing_percent[missing_percent > 0.5].index
df.drop(columns=columns_to_drop, inplace=True)
print(f"\n🗑 ستون‌های حذف‌شده: {list(columns_to_drop)}")

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
object_cols = df.select_dtypes(include=['object']).columns

imputer = KNNImputer(n_neighbors=5)
df_numeric = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)

df_object = df[object_cols].copy()
for col in df_object.columns:
    df_object[col] = df_object[col].fillna(df_object[col].mode()[0])

df_cleaned = pd.concat([df_numeric, df_object], axis=1)

z_scores = np.abs(zscore(df_cleaned[numeric_cols]))
df_final = df_cleaned[(z_scores < 3).all(axis=1)]
print(f"\n📉 Outlierها حذف شدند. شکل نهایی: {df_final.shape}")
print(df_final.head(5))

desc_stats = df_final[numeric_cols].describe().loc[['min', 'max', 'std']]
print("\n📈 آمار توصیفی:")
print(desc_stats)

correlation_matrix = df_final.corr(numeric_only=True)
plt.figure(figsize=(16, 14), dpi=100)
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt=".2f", linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix of Numerical Features", fontsize=18, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

corr_with_price = correlation_matrix['SalePrice'].drop('SalePrice')
top_features = corr_with_price.abs().sort_values(ascending=False).head(10)
print("📌 10 ویژگی با بیشترین همبستگی با SalePrice:")
print(top_features)

for feature in top_features.index:
    g = sns.jointplot(data=df_final, x=feature, y='SalePrice', kind='reg', height=7, space=0.2, marginal_kws=dict(bins=30, fill=True))
    plt.suptitle(f"Relationship Between {feature} and SalePrice", fontsize=14, y=1.02)
    g.ax_joint.set_xlabel(feature, fontsize=12)
    g.ax_joint.set_ylabel("SalePrice", fontsize=12)
    plt.tight_layout()
    plt.show()

X_full = df_final.select_dtypes(include=['float64', 'int64']).drop(columns=['SalePrice'])
y_full = df_final['SalePrice']

selector = SelectKBest(score_func=f_regression, k=10)
selector.fit(X_full, y_full)

selected_mask = selector.get_support()
selected_features = X_full.columns[selected_mask]
scores = selector.scores_[selected_mask]

selected_df = pd.DataFrame({
    "Feature": selected_features,
    "F-score": scores
}).sort_values(by="F-score", ascending=False)

print("\n📌 ویژگی‌های انتخاب‌شده (مرتب‌شده بر اساس F-score):")
print(selected_df)

plt.figure(figsize=(10, 6), dpi=120)
plt.barh(selected_df["Feature"], selected_df["F-score"], color='teal')
plt.xlabel("F-score")
plt.title("Top 10 Features by F-score (SelectKBest)")
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

X_selected = X_full[selected_df["Feature"]]
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_full, test_size=0.25, random_state=42
)

print(f"\n✅ Train shape: {X_train.shape}, Test shape: {X_test.shape}")

def evaluate_model(model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return {"Model": name, "RMSE": rmse, "R² Score": r2}

results = []
results.append(evaluate_model(LinearRegression(), "Linear Regression"))
results.append(evaluate_model(Lasso(alpha=1.0), "Lasso Regression"))
results.append(evaluate_model(Ridge(alpha=1.0), "Ridge Regression"))
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
results.append(evaluate_model(poly_model, "Polynomial Regression (deg=2)"))

results_df = pd.DataFrame(results).sort_values(by="RMSE")
print("\n📊 مقایسه مدل‌ها (مرتب‌شده بر اساس RMSE):")
print(results_df)

best_model_name = results_df.iloc[0]["Model"]
if best_model_name == "Linear Regression":
    best_model = LinearRegression()
elif best_model_name == "Lasso Regression":
    best_model = Lasso(alpha=1.0)
elif best_model_name == "Ridge Regression":
    best_model = Ridge(alpha=1.0)
elif "Polynomial" in best_model_name:
    best_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_best, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title(f"Actual vs Predicted - {best_model_name}")
plt.grid(True)
plt.tight_layout()
plt.show()

residuals = y_test - y_pred_best
plt.figure(figsize=(8, 5))
plt.scatter(y_pred_best, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted SalePrice")
plt.ylabel("Residual")
plt.title(f"Residual Plot - {best_model_name}")
plt.grid(True)
plt.tight_layout()
plt.show()

cv_scores = cross_val_score(best_model, X_selected, y_full, cv=5, scoring="neg_root_mean_squared_error")
cv_rmse_scores = -cv_scores

print(f"\n📉 Cross-Validation RMSE scores (5-fold) for {best_model_name}:")
print(cv_rmse_scores)
print(f"📊 Mean RMSE: {cv_rmse_scores.mean():.2f} | Std: {cv_rmse_scores.std():.2f}")

plt.figure(figsize=(6, 5), dpi=120)
plt.boxplot(cv_rmse_scores, vert=True, patch_artist=True, boxprops=dict(facecolor="lightblue"))
plt.title("Cross-Validation RMSE Scores (5-Fold)")
plt.ylabel("RMSE")
plt.xticks([1], ["Polynomial Regression (deg=2)"])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
