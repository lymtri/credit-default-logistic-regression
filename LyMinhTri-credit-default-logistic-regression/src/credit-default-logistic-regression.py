# -*- coding: utf-8 -*-
import pandas as pd

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "..", "data", "UCI_Credit_Card.csv")

df = pd.read_csv(csv_path)

df.head()

print(df.info())
print(df.describe())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# ==========================================
# 1. LOAD DỮ LIỆU
# ==========================================
# df = pd.read_csv("default_credit_card.csv")

# ==========================================
# 2. TIỀN XỬ LÝ LOGIC
# ==========================================
df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})

# PAY_x: clip giá trị âm về 0 để giữ tính ordinal
pay_cols = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
for col in pay_cols:
    df[col] = df[col].clip(lower=0)

# ==========================================
# 3. TÁCH FEATURE / TARGET
# ==========================================
X = df.drop(['ID', 'default'], axis=1)
y = df['default']

# ==========================================
# 4. CHIA TRAIN / TEST
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 5. COLUMN TRANSFORMER
# ==========================================
categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ]
)

# ==========================================
# 6. PIPELINE LOGISTIC REGRESSION
# ==========================================
model = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('logreg', LogisticRegression(
        class_weight={0: 1, 1: 3},
        max_iter=1000,
        random_state=42
    ))
])

# ==========================================
# 7. TRAIN MODEL
# ==========================================
model.fit(X_train, y_train)

# ==========================================
# 8. DỰ ĐOÁN & ĐÁNH GIÁ
# ==========================================
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# 9. TRÍCH XUẤT HỆ SỐ (BETA + ODDS RATIO)
# ==========================================
# Lấy tên cột sau one-hot
ohe = model.named_steps['preprocess'].named_transformers_['cat']
ohe_features = ohe.get_feature_names_out(categorical_cols)

all_features = numeric_cols + list(ohe_features)

coef = model.named_steps['logreg'].coef_[0]

importance = pd.DataFrame({
    'Feature': all_features,
    'Beta': coef,
    'Odds Ratio': np.exp(coef)
}).sort_values(by='Beta', ascending=False)

print("\nTop 5 biến ảnh hưởng mạnh nhất:")
print(importance.head(5))

import seaborn as sns
import matplotlib.pyplot as plt

df_sex = df.groupby('SEX')['default'].mean().reset_index()

plt.figure(figsize=(8,5))
sns.barplot(
    x='SEX',
    y='default',
    data=df_sex,
    color='#1f77b4',         # màu xanh siêu nhạt
    edgecolor='white',       # viền trắng để gần như biến mất
    linewidth=0.3            # viền cực mỏng
)

plt.title('Tỷ lệ default theo giới tính')
plt.xlabel('Giới tính')
plt.ylabel('Tỷ lệ default')

plt.xticks([0,1], ['Nam', 'Nữ'])
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

df_geo = df.groupby('EDUCATION')['default'].mean().reset_index()

plt.figure(figsize=(8,5))
sns.barplot(
    x='EDUCATION',
    y='default',
    data=df_geo,
    color='#1f77b4',        # màu xanh siêu nhạt (Light Blue)
    edgecolor='white',      # viền trắng để gần như biến mất
    linewidth=0.3           # viền cực mỏng
)

plt.title('Tỷ lệ default theo nhóm khách hàng (trình độ học vấn)')
plt.xlabel('Nhóm khách hàng')
plt.ylabel('Tỷ lệ default')

plt.xticks([0,1,2,3], ['Sau ĐH', 'ĐH', 'THPT', 'Khác'])
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

sns.kdeplot(
    data=df,
    x='LIMIT_BAL',
    hue='default',
    fill=True,
    alpha=0.5
)

plt.title('Phân phối hạn mức tín dụng theo tình trạng vỡ nợ')
plt.xlabel('Hạn mức tín dụng (LIMIT_BAL)')
plt.ylabel('Mật độ')

plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

sns.kdeplot(
    data=df,
    x='AGE',
    hue='default',
    fill=True,
    alpha=0.5,
    palette={0: "#87CEFA", 1: "orange"}  # đổi màu ở đây
)

plt.title('Phân phối AGE theo default')
plt.xlabel('AGE')
plt.ylabel('Mật độ')

plt.show()

df.groupby('default')['AGE'].describe()

import seaborn as sns
import matplotlib.pyplot as plt

# KDE plot cho LIMIT_BAL theo tình trạng default
plt.figure(figsize=(8,6))


sns.kdeplot(
    data=df,
    x='LIMIT_BAL',
    hue='default',
    fill=True,
    alpha=0.5,
    palette={0: "#87CEFA", 1: "orange"}  # đổi màu ở đây
)

plt.title("KDE Plot: Phân phối LIMIT_BAL theo tình trạng default")
plt.xlabel("LIMIT_BAL")
plt.ylabel("Density")
plt.show()

df.groupby('default')['LIMIT_BAL'].describe()

import seaborn as sns
import matplotlib.pyplot as plt

# Lọc dữ liệu PAY_AMT1 <= 20000
df_filtered = df[df["PAY_AMT1"] <= 20000]

sns.set_style("whitegrid")
sns.set_context("talk")

plt.figure(figsize=(9,7))
sns.violinplot(
    data=df_filtered,
    x="default",
    y="PAY_AMT1",
    density_norm="width",           # thay cho scale="width"
    inner="quartile",
    linewidth=1.5,
    hue="default",
    palette={0:"#87CEFA", 1:"orange"},
    legend=False
)

plt.title("Violin Plot: PAY_AMT1 theo Default", fontsize=16, fontweight="bold", pad=15)
plt.xlabel("Default", fontsize=14)
plt.ylabel("PAY_AMT1", fontsize=14)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

df.groupby("default")["PAY_AMT1"].describe()

import pandas as pd
import matplotlib.pyplot as plt

# Giả sử df có cột PAY_0 và default
# Lọc dữ liệu chỉ lấy PAY_0 từ 0 đến 8
df_filtered = df[df['PAY_0'].between(0, 8)]

# Tính tỷ lệ vỡ nợ theo từng mức PAY_0
ratio_by_pay0 = df_filtered.groupby('PAY_0')['default'].mean()

# Vẽ biểu đồ cột
plt.figure(figsize=(8,6))
ratio_by_pay0.plot(kind='bar', color='steelblue', edgecolor='black')

plt.title('Tỷ lệ default theo PAY_0 (0–8)')
plt.xlabel('PAY_0 (Mức độ chậm trả kỳ gần nhất)')
plt.ylabel('Tỷ lệ default')
plt.xticks(rotation=0)
plt.ylim(0,1)

# Hiển thị giá trị trên cột
for idx, val in enumerate(ratio_by_pay0):
    plt.text(idx, val + 0.02, f"{val:.2f}", ha='center')

plt.tight_layout()
plt.show()

import pandas as pd

# Giả sử df là DataFrame với cột 'PAY_0' và 'default' (1 = vỡ nợ, 0 = không vỡ nợ)

# Tính số lượng khách hàng theo từng mức PAY_0
count_by_pay0 = df['PAY_0'].value_counts().sort_index()

# Tính tỷ lệ vỡ nợ theo từng mức PAY_0
ratio_by_pay0 = df.groupby('PAY_0')['default'].mean()

# In kết quả
print("Số lượng khách hàng theo PAY_0:")
print(count_by_pay0)
print("\nTỷ lệ vỡ nợ theo PAY_0:")
print(ratio_by_pay0)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Tính các metric với zero_division
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
roc_auc   = roc_auc_score(y_test, y_pred_proba)

import pandas as pd
metrics = {
    "Metric": ["ROC AUC", "Accuracy", "Precision", "Recall", "F1-score"],
    "Giá trị": [roc_auc, accuracy, precision, recall, f1]
}
results_df = pd.DataFrame(metrics)
print(results_df)

from sklearn.metrics import confusion_matrix


cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

print("Bảng 5.2: Confusion Matrix")
print(f"\t\tDự đoán: 0 (Thanh toán đúng hạn)\tDự đoán: 1 (Vỡ nợ)")
print(f"Thực tế: 0 (Thanh toán đúng hạn)\tTN = {TN}\t\tFP = {FP}")
print(f"Thực tế: 1 (Vỡ nợ)\tFN = {FN}\t\tTP = {TP}")

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0,1], [0,1], color='red', linestyle='--', label='Baseline (random)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Biểu đồ ROC')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['non-default', 'default'],
            yticklabels=['non-default', 'default'])

plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Confusion Matrix')
plt.show()

import statsmodels.api as sm

# 1. Fit preprocessor trên X_train (nếu chưa fit riêng)
preprocessor.fit(X_train)

# 2. Biến đổi X_train bằng preprocessor
X_train_transformed = preprocessor.transform(X_train)

# 3. Lấy tên cột sau biến đổi
ohe = preprocessor.named_transformers_['cat']
ohe_features = ohe.get_feature_names_out(categorical_cols)
all_features = numeric_cols + list(ohe_features)

# 4. Đưa vào DataFrame để dùng với statsmodels
X_train_final = pd.DataFrame(X_train_transformed, columns=all_features).astype(float)
y_train_final = y_train.astype(float).values

# 5. Thêm hằng số
X_train_sm = sm.add_constant(X_train_final)

# 6. Chạy Logit
sm_model = sm.Logit(y_train_final, X_train_sm)
result = sm_model.fit(method='newton', max_iter=5000, disp=False)

# 7. Trích xuất OR như bạn làm
coefficients = result.params
p_values = result.pvalues
conf_int = result.conf_int(alpha=0.05)

or_table = pd.DataFrame({
    'Feature': coefficients.index,
    'OR': np.exp(coefficients.values),
    '95% CI': [
        '[{:.4f}, {:.4f}]'.format(np.exp(conf_int.iloc[i, 0]), np.exp(conf_int.iloc[i, 1]))
        for i in range(len(coefficients))
    ],
    'p-value': p_values.values
}).sort_values(by='p-value').reset_index(drop=True)

print(or_table.round(4))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Lọc bỏ hằng số từ or_table
plot_data = or_table[or_table['Feature'] != 'const'].copy()

# Lấy các giá trị CI số học tương ứng với các feature trong plot_data, đảm bảo đúng thứ tự
# `conf_int` là DataFrame chứa các khoảng tin cậy của hệ số từ `statsmodels.Logit`.
# `np.exp()` để chuyển đổi từ CI của hệ số sang CI của Odds Ratio.
ci_numerical = np.exp(conf_int.loc[plot_data['Feature']])

plot_data['Lower 95% CI_numerical'] = ci_numerical[0].values
plot_data['Upper 95% CI_numerical'] = ci_numerical[1].values

# Sắp xếp plot_data theo OR để biểu đồ dễ đọc hơn
plot_data = plot_data.sort_values(by='OR', ascending=True).reset_index(drop=True)

plt.figure(figsize=(10, len(plot_data) * 0.4 + 2))

# Tính toán xerr cho cả hai phía: từ OR đến cận dưới và từ OR đến cận trên
xerr_lower = plot_data['OR'] - plot_data['Lower 95% CI_numerical']
xerr_upper = plot_data['Upper 95% CI_numerical'] - plot_data['OR']

plt.errorbar(
    x=plot_data['OR'],
    y=plot_data['Feature'],
    xerr=[xerr_lower, xerr_upper], # Sử dụng cả cận dưới và cận trên cho error bars
    fmt='o', color='royalblue', ecolor='lightgray', capsize=4
)

plt.axvline(x=1, color='red', linestyle='--', label='OR = 1 (No effect)')
plt.xlabel('Odds Ratio')
plt.title('Biểu đồ Forest Plot: Tỷ lệ Chênh lệch (Odds Ratios) & Khoảng tin cậy 95%')
plt.grid(True, ls="-.", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()