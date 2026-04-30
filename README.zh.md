# Agent Trajectory Eval

一个 Codex-first 的编程 Agent 轨迹评测工具。它读取 Codex CLI JSONL
会话或 `codex exec --json` 输出流，并生成：

1. **归一化步骤**：把 Codex 事件转换成统一步骤记录
2. **轨迹树**：展示由文件/环境变更触发的状态迁移
3. **风险指标**：统计命令、测试、文件变更、失败测试和可疑步骤
4. **故障诊断**：基于规则定位关键步骤，并给出 replay 建议
5. **批量报告**：支持目录级评测汇总，便于 CI 或回归分析

当前项目只默认适配 Codex JSONL。适配器接口仍然保留，但默认注册的只有
Codex adapter。

## 快速开始

```bash
trace-agent eval --input examples/codex_failed_run_001.jsonl --output out/codex_eval
```

评测一个目录：

```bash
trace-agent eval --input data/lcb/trajectories --output out/lcb_eval
```

CI 风格退出码：

```bash
trace-agent eval --input examples/codex_failed_run_001.jsonl --output out/codex_eval --ci
```

- `0`：工具运行成功，且没有失败或中/高风险轨迹
- `1`：评测成功，但至少一个轨迹失败或达到中/高风险
- `2`：工具错误、输入非法或格式不支持

旧的平铺参数形式仍然兼容：

```bash
trace-eval --input examples/codex_failed_run_001.jsonl --output out/codex_eval
```

## 命令

```bash
# 拉取一小组 LiveCodeBench 题目到 data/lcb/problems
trace-agent lcb fetch

# 调用 Codex 跑一道 easy 题，并保存 JSONL 轨迹
trace-agent lcb run --difficulty easy --limit 1

# 评测生成的 LiveCodeBench 轨迹
trace-agent lcb eval
```

如果是在源码目录里直接运行、还没安装包，可以在前面加 `uv run`，例如：
`uv run trace-agent eval --input ... --output ...`。

## 输出文件

单条轨迹输出目录包含：

- `normalized_steps.json`
- `trace_tree.md`
- `diagnosis.json`
- `diagnosis.md`
- `eval_result.json`
- `eval_summary.md`

批量评测还会生成：

- `batch_summary.json`
- `batch_summary.md`

## 输入格式

当前支持 Codex JSONL：每行一个 JSON event。评测器会处理
`thread.started`、`turn.completed`、`turn.failed` 和 `item.completed`，并识别
Codex item 类型，例如 `reasoning`、`command_execution`、`file_change`、
`agent_message`、`mcp_tool_call`、`error` 和 `web_search`。

仓库里的旧内部 JSON 示例仅作为历史样本保留，默认评测流程现在是 Codex-only。

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

- `models.py`: 数据结构 (Trajectory, Step, NormalizedStep, TraceNode, Diagnosis, EvalResult)
- `parser.py`: Codex JSONL 加载和验证
- `evaluator.py`: 单文件评测、目录发现和批量汇总
- `adapters/codex_adapter.py`: Codex 事件流转换
- `classifier.py`: 动作类型、阶段和状态变更分类
- `tree.py`: 轨迹树构建和渲染
- `analyzer.py`: 可疑步骤评分和故障诊断
- `report.py`: 输出文件生成
- `main.py`: CLI 接口

## 技术细节

- **语言**: Python 3.x
- **依赖**: 仅 Python 标准库
- **设计**: 职责分离，易于扩展
- **测试**: `python -m unittest discover`

## 扩展分析器

扩展方式：

1. 在 `classifier.py` 中添加新的动作类型
2. 在 `analyzer.py` 中添加新的可疑规则
3. 在 `report.py` 中增强输出格式
4. 在 `tree.py` 中修改树渲染逻辑
