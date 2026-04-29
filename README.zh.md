# Codex Trajectory Analyzer

一个用于分析编程 Agent 执行轨迹的轻量级 Python 命令行工具。解析 Agent 的执行记录并生成：

1. **归一化步骤**: 带动作类型和阶段标注的增强步骤数据
2. **轨迹树**: Agent 探索过程和状态变更的可视化展示
3. **可疑步骤检测**: 对潜在问题模式进行评分
4. **故障诊断**: 定位关键故障点并提供回放建议

## 输入格式

工具需要一个如下结构的 JSON 文件：

```json
{
  "task": "Fix the parser bug",
  "final_status": "failed",
  "steps": [
    {
      "step_id": 1,
      "thought": "I need to inspect the parser",
      "action": "rg \"parse\" .",
      "observation": "parser.py contains parse_config",
      "diff": null
    }
  ]
}
```

### 必填字段
- `task`: Agent 要完成的任务描述
- `final_status`: "success" 或 "failed"
- `steps`: 步骤对象数组

### 步骤字段
- `step_id` (必填): 整数标识符
- `thought` (可选): Agent 的推理过程
- `action` (必填): Agent 执行的命令
- `observation` (可选): 执行结果
- `diff` (可选): 文件变更内容

## 运行方式

```bash
python main.py --input examples/failed_run_001.json --output out/failed_run_001
```

会在输出目录生成四个文件：

- `normalized_steps.json` - 带分类信息的完整步骤数据
- `trace_tree.md` - Agent 执行的可视化树
- `diagnosis.json` - 机器可读的诊断结果
- `diagnosis.md` - 人类可读的诊断报告

## 核心概念

### 探索 vs 状态变更

- **探索 (Explore)**: 收集信息但不修改系统的步骤（搜索、检查）
- **状态变更 (State Change)**: 修改系统的步骤（文件编辑、环境变更）

状态变更步骤在轨迹树中创建新的"状态"。Agent 在一个状态内探索，做出变更后转换到新状态。

### 阶段 (Stages)

动作被分类到以下阶段：
- **environment verification**: 检查工具版本和环境配置
- **dependency installation**: 安装包或依赖
- **inspection/debugging**: 搜索和检查文件
- **patching**: 修改代码
- **verification**: 运行测试
- **other**: 不属于其他分类的动作

### 动作类型 (Action Types)

- `inspect_file`: 读取文件内容 (cat, sed, head, tail)
- `search`: 搜索代码 (rg, grep, find)
- `run_test`: 运行测试 (pytest, cargo test, npm test)
- `edit_file`: 修改文件 (apply_patch, 写文件)
- `env_change`: 安装依赖 (pip install, npm install)
- `git_action`: Git 操作
- `other`: 未分类的动作

## 可疑步骤检测

工具检测以下潜在问题模式：

- **测试文件篡改**: 编辑测试文件使测试通过
- **补丁后测试仍失败**: 修改未能修复问题
- **重复命令**: 冗余操作
- **重复测试失败**: 同一测试反复失败且无干预
- **环境问题**: 环境变更后的依赖问题
- **Git 回滚**: 试错行为

每个可疑步骤获得一个评分 (0.0 到 1.0+) 和解释性原因。

## 输出文件

### normalized_steps.json

带所有分类信息的完整步骤数据：

```json
[
  {
    "step_id": 1,
    "thought": "I need to inspect the parser",
    "action": "rg \"parse\" .",
    "observation": "parser.py contains parse_config",
    "diff": null,
    "action_type": "search",
    "stage": "inspection/debugging",
    "state_change": false,
    "suspicious_score": 0.0,
    "suspicious_reasons": []
  }
]
```

### trace_tree.md

展示状态转换的可视化表示：

```markdown
# Trace Tree

State 0
  - Step 1 [inspection/debugging | search | explore] rg "parse" .
  - Step 2 [inspection/debugging | inspect_file | explore] sed -n '1,160p' parser.py
  - Step 3 [patching | edit_file | state_change] apply_patch parser.py
    -> State 1
```

### diagnosis.md

人类可读的分析报告，包含：
- 任务描述和最终状态
- 关键故障步骤
- 所有可疑步骤的评分和原因表
- 回放建议和替代方案提示

## 局限性

这是一个**最小可行产品 (MVP)** — 基于规则的分析器，而非完整的 CodeTracer 实现。使用简单的模式匹配和启发式规则，而非机器学习或复杂的语义分析。

## 架构

代码按职责分为清晰的模块：

- `models.py`: 数据结构 (Step, NormalizedStep, TraceNode, Diagnosis)
- `parser.py`: JSON 加载和验证
- `classifier.py`: 动作类型、阶段和状态变更分类
- `tree.py`: 轨迹树构建和渲染
- `analyzer.py`: 可疑步骤评分和故障诊断
- `report.py`: 输出文件生成
- `main.py`: CLI 接口

## 技术细节

- **语言**: Python 3.x
- **依赖**: 仅 Python 标准库
- **设计**: 职责分离，易于扩展

## 扩展分析器

扩展方式：

1. 在 `classifier.py` 中添加新的动作类型
2. 在 `analyzer.py` 中添加新的可疑规则
3. 在 `report.py` 中增强输出格式
4. 在 `tree.py` 中修改树渲染逻辑
