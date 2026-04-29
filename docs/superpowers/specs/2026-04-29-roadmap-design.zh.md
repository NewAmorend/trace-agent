# Codex Trajectory Analyzer — 完善计划设计

## 目标

将单格式的 CLI 工具转型为一个可扩展的开源轨迹分析器，专注于编程 Agent 的故障诊断。

## 原则

- 核心使用纯 Python 标准库；可选依赖（pytest、LLM provider）仅用于开发和扩展
- 使用适配器模式支持多格式；通过 hook 支持未来扩展
- CLI 优先，不做 Web UI 或服务端组件

## 优先级顺序

### 1. 适配器层重构

**问题**: `parser.py` 直接解析单一 JSON 格式，添加新格式需要修改核心代码。

**设计**:

```
adapters/
├── __init__.py        # 注册机制 + get_adapter()
├── base.py            # 适配器抽象基类
├── codex_adapter.py   # OpenAI Codex 轨迹格式
└── internal.py        # 当前内部 JSON 格式
```

**接口**:

```python
class BaseAdapter(ABC):
    @abstractmethod
    def detect(self, data: dict) -> bool:
        """判断此适配器是否能处理给定的 JSON 结构。"""

    @abstractmethod
    def transform(self, data: dict) -> tuple[str, str, list[Step]]:
        """转换为统一的 (task, final_status, steps) 格式。"""
```

**流程**: `main.py` 加载 JSON，遍历已注册的适配器，调用 `detect()` 后再调用 `transform()`。当前格式迁入 `internal.py`，行为不变。

**Codex 适配器**: 解析 Codex 会话格式，将其字段映射到 `Step` 数据类。具体字段映射待获取 Codex 样本轨迹后确定。

### 2. 项目工程化

- `pyproject.toml` — 包配置，支持 `pip install` 后使用 `trace-analyzer` 命令
- `requirements-dev.txt` — pytest 及开发工具
- `LICENSE` — MIT 许可证
- `README.md` — 安装、使用示例、贡献指南
- `CONTRIBUTING.md` — 如何添加适配器、模式和运行测试

### 3. 测试套件 (pytest)

```
tests/
├── conftest.py           # 共享 fixtures
├── test_adapters.py      # 适配器检测与转换
├── test_classifier.py    # 动作分类、阶段逻辑
├── test_tree.py          # 树构建和渲染
├── test_analyzer.py      # 可疑评分、故障定位
├── test_report.py        # 输出文件生成
└── fixtures/
    ├── sample_internal.json
    └── sample_codex.json
```

- 在添加新功能前先对现有模块实现全覆盖
- 内部格式和 Codex 格式的 fixture 文件
- pytest 作为开发依赖

### 4. 分类器 LLM Hook

**问题**: 基于规则的分类有局限性；用户可能需要基于 LLM 的评判。

**设计**: 在 `classifier.py` 中添加 hook 点：

```python
class StepClassifier:
    def __init__(self, judge=None):
        # judge=None 使用默认的基于规则的分类
        # judge=<callable> 委托给外部分类器
        self._judge = judge

    def classify(self, step: Step) -> NormalizedStep:
        if self._judge:
            return self._judge(step)
        return self._rule_based_classify(step)
```

- 默认行为不变（基于规则）
- Hook 接受任何接收 `Step` 并返回 `NormalizedStep` 的可调用对象
- 不包含特定 provider 的代码；用户自行接入 LLM 调用

### 5. 诊断增强

**模式库**: 将 `analyzer.py` 中硬编码的可疑模式提取到配置文件中（`patterns.yaml` 或 `patterns.json`）：

```yaml
patterns:
  - name: test_file_manipulation
    description: "Agent 在执行过程中修改了测试文件"
    indicators:
      - action_type: edit_file
        path_pattern: "*/test_*"
    score_weight: 0.8
```

**置信度分级**: 每条诊断带置信度（高/中/低），基于匹配的模式数量和权重计算。

**修复建议**: 诊断输出包含基于匹配模式的修复建议。

### 6. 批量分析

- `--input` 支持传入目录路径（除单文件外）
- 扫描所有 `.json` 文件，逐个分析
- 生成汇总报告：成功/失败率、常见失败模式排行、耗时分布
- 单条轨迹分析功能不变

### 7. CI 集成

- 退出码：0 = 未发现问题，1 = 诊断发现问题，2 = 工具错误
- `--format json` 标志：结构化输出，方便 CI 流水线消费
- `--quiet` 标志：抑制进度输出，只打印结论

## 不在范围内

- Web UI 或服务端模式
- 实时监控
- 多 Agent 编排
- 数据库存储
