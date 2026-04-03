import pandas as pd

df = pd.read_excel(
    "data/hiera_prediction_results_20260310_200800.xlsx", sheet_name="类别统计"
)
print("类别统计完整内容:")
print(df.to_string())
print("\n列名:")
print(df.columns.tolist())
print("\n前10行数据:")
for i in range(min(10, len(df))):
    print(f"Row {i}:")
    print(f"  类别ID: {df.iloc[i]['类别ID']}")
    print(f"  类别名称: {df.iloc[i]['类别名称']}")
    print(f"  正确预测数: {df.iloc[i]['正确预测数']}")
    print(f"  总样本数: {df.iloc[i]['总样本数']}")
    print(f"  错误预测数: {df.iloc[i]['错误预测数']}")
    accuracy_col = df.columns[5]
    print(f"  准确率(%): {df.iloc[i][accuracy_col]}")
    print()
