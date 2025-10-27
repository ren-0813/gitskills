import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score,
                             mean_squared_error, r2_score, confusion_matrix)

# 设置中文字体，解决显示问题
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


# ------------------------------------------------------
# 1. 数据加载与列名确认
# ------------------------------------------------------
# 加载数据集（请替换为实际文件路径）
df = pd.read_csv("health_lifestyle_dataset.csv")

# 显示数据集所有列名（关键步骤：用于确认实际列名）
print("数据集列名：", df.columns.tolist())
print(f"\n数据集形状：{df.shape}")
print("\n前5行数据预览：")
print(df.head())

# ------------------------------------------------------
# 2. 数据预处理
# ------------------------------------------------------
# 检查缺失值
print("\n缺失值统计：")
print(df.isnull().sum())

# 分离数值列和类别列
numeric_cols = ['age', 'bmi', 'daily_steps', 'sleep_hours', 'water_intake_l', 'calories_consumed','resting_hr','systolic_bp', 'diastolic_bp', 'cholesterol']
categorical_cols = ['gender','smoker', 'alcohol', 'family_history', 'disease_risk']

# 填充缺失值
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ------------------------------------------------------
# 3. 探索性数据分析(EDA)
# ------------------------------------------------------
# 3.1 数值特征相关性分析
plt.figure(figsize=(12, 8))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('数值特征相关性矩阵')
plt.tight_layout()
plt.show()

# 3.2 生活方式与健康指标关系 - 运动（以daily_steps为例）与BMI
plt.figure(figsize=(10, 6))
sns.boxplot(x=pd.cut(df['daily_steps'], bins=5), y='bmi', data=df)
plt.title('不同步数区间人群的BMI分布')
plt.xlabel('每日步数区间')
plt.ylabel('BMI指数')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3.3 睡眠与血压关系
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sleep_hours', y='systolic_bp',
                hue='disease_risk', size='age', data=df, alpha=0.7)
plt.title('睡眠时长与收缩压的关系（按疾病风险和年龄分组）')
plt.xlabel('睡眠时长（小时）')
plt.ylabel('收缩压（mmHg）')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ------------------------------------------------------
# 4. 建模准备（动态定义目标变量）
# ------------------------------------------------------
# 分类任务目标变量：构建高血压标签
df['Hypertension'] = ((df['systolic_bp'] > 140) |
                      (df['diastolic_bp'] > 90)).astype(int)
y_class = df['Hypertension']

# 回归任务目标变量：以cholesterol为例
y_reg = df['cholesterol']

# 特征列（排除目标变量）
X = df.drop(['Hypertension', 'cholesterol'], axis=1)

# 区分数值和类别特征
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# 预处理管道
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

# 划分训练集和测试集
X_train, X_test, y_train_class, y_test_class = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

# ------------------------------------------------------
# 5. 分类任务建模
# ------------------------------------------------------
print(f"\n===== 分类任务：预测是否患有高血压 =====")
clf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

clf_pipeline.fit(X_train, y_train_class)
y_pred = clf_pipeline.predict(X_test)
y_pred_proba = clf_pipeline.predict_proba(X_test)[:, 1]

# 评估
print(f"准确率: {accuracy_score(y_test_class, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test_class, y_pred_proba):.4f}")
print("\n分类报告:")
print(classification_report(y_test_class, y_pred))

# 混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test_class, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('预测结果')
plt.ylabel('实际结果')
plt.title('高血压预测混淆矩阵')
plt.tight_layout()
plt.show()

# 特征重要性
cat_features = list(clf_pipeline.named_steps['preprocessor']
                   .named_transformers_['cat']
                   .get_feature_names_out(categorical_features))
all_features = numeric_features + cat_features
importances = clf_pipeline.named_steps['classifier'].feature_importances_

top_features = pd.DataFrame({
    '特征': all_features,
    '重要性': importances
}).sort_values('重要性', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='重要性', y='特征', data=top_features)
plt.title('分类任务Top10特征重要性')
plt.tight_layout()
plt.show()

# ------------------------------------------------------
# 6. 回归任务建模
# ------------------------------------------------------
print(f"\n===== 回归任务：预测胆固醇水平 =====")
reg_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

reg_pipeline.fit(X_train_reg, y_train_reg)
y_pred_reg = reg_pipeline.predict(X_test_reg)

# 评估
print(f"MSE: {mean_squared_error(y_test_reg, y_pred_reg):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)):.4f}")
print(f"R²分数: {r2_score(y_test_reg, y_pred_reg):.4f}")

# 预测值vs真实值
plt.figure(figsize=(10, 6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.6)
plt.plot([y_test_reg.min(), y_test_reg.max()],
         [y_test_reg.min(), y_test_reg.max()], 'r--')
plt.xlabel('真实胆固醇水平')
plt.ylabel('预测胆固醇水平')
plt.title('胆固醇预测值 vs 真实值')
plt.tight_layout()
plt.show()

print("\n分析完成！所有步骤已适配数据集实际列名。")