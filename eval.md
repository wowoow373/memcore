# memcore 实验完整分析报告

## 1. 实验设置

- **模型**: GLM-4.7-flash（agent + user simulator 均使用）
- **领域**: tau-bench airline test set (33 tasks)
- **任务子集**: GPT-4o 能过但 GLM baseline 失败的 18 个任务（来自 baseline analysis 的 gpt4o_only 列表）
- **Baseline**: GLM-4.7-flash + 标准 tool-calling agent
- **Seed Guidance**: 在每次 tool call 后，根据 action name 查找预先写好的推理要点由dsv4根据gpt4o轨迹撰写，实际可以是每次teacher依据teacher模型成功的轨迹或者人类成功轨迹生成，作为 `expert_guidance` tool result 注入到对话历史中
- **任务 ID 列表**: 2, 5, 7, 11, 17, 20, 27, 34, 35, 36, 37, 39, 42, 44, 45, 46, 47, 48

## 2. 逐任务对比

```
Task   Baseline     Seed       
────   ────────     ────       
[ 2]   FAIL         PASS       
[ 5]   FAIL         FAIL       
[ 7]   FAIL         FAIL       
[11]   FAIL         FAIL       
[17]   FAIL         FAIL       
[20]   FAIL         FAIL       
[27]   FAIL         FAIL       
[34]   FAIL         FAIL       
[35]   FAIL         FAIL         
[36]   FAIL         FAIL         
[37]   FAIL         FAIL       
[39]   FAIL         FAIL       
[42]   FAIL         PASS          
[44]   FAIL         PASS         
[45]   FAIL         FAIL         
[46]   FAIL         FAIL       
[47]   FAIL         PASS      
[48]   FAIL         PASS       
```

### 通过率

- Seed Guidance 总体: **5/18 = 27.8%**
- 则在gpt4o上通过率提升：**15/33 = 45%   ->    20/33 = 61%** 

