import pandas as pd
import numpy as np
import os

# 1. 创建 data 目录
os.makedirs('data', exist_ok=True)

# 2. 生成模拟交通数据：1000行，16个特征
# 对应 config.yaml 中的 input_dim: 16
data = np.random.rand(1000, 16)
columns = [f'feature_{i}' for i in range(16)]
columns[0] = 'energy_consumption' # 设定目标列

df = pd.DataFrame(data, columns=columns)
df.to_csv('data/transport_data.csv', index=False)
print("✅ Mock data generated at data/transport_data.csv")

# 3. 简单验证模型构建
from src.models.model_builder import build_optimization_model
config = {
    "input_dim": 16,
    "hidden_units": [256, 128, 64],
    "dropout_rate": 0.3
}
model = build_optimization_model(config)
print("✅ Model built successfully!")
model.summary()