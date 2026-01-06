# ArXiv AI/算法论文每日更新

这是一个自动获取、分析和整理 ArXiv AI/算法领域最新论文的系统，具有 AI 驱动的分类功能。

## 功能特点

- **自动论文获取**：自动从 ArXiv 获取最新 AI/算法论文
- **AI 驱动分析**：使用豆包（Doubao）大模型进行智能论文分类和分析
- **双语支持**：提供中英文论文标题
- **代码链接检测**：自动提取 GitHub 仓库链接
- **结构化输出**：生成结构良好的 Markdown 报告
- **并行处理**：使用多线程提高效率
- **智能分类**：将论文分类到特定研究领域
- **核心贡献提取**：使用 LLM 提取论文的核心贡献

## 项目结构

```
ArXiv_AI_Papers_Daily/
├── scripts/           # 脚本文件
│   ├── get_ai_papers.py       # 主程序
│   ├── ai_categories_config.py # AI/算法分类配置
│   ├── llm_helper.py          # LLM API 助手
│   ├── doubao_client.py       # 豆包 API 客户端
│   ├── config.py              # 配置文件
│   └── requirements.txt      # 依赖项
├── data/              # 表格格式论文信息
│   └── YYYY-MM/
│       └── YYYY-MM-DD.md
└── local/             # 详细格式论文信息
    └── YYYY-MM/
        └── YYYY-MM-DD.md
```

## 系统要求

- Python 3.x
- requirements.txt 中列出的依赖项
- 豆包（Doubao）API 密钥
- 稳定的网络连接

## 安装方法

1. 克隆仓库
2. 安装依赖：
```bash
pip install -r requirements.txt
```
3. 配置豆包 API 密钥（在 `config.py` 中）

## 配置说明

主要参数（在 `get_ai_papers.py` 中）：
- `QUERY_DAYS_AGO`：查询几天前的论文（0=今天，1=昨天）
- `MAX_RESULTS`：最大返回论文数量
- `MAX_WORKERS`：并行处理的最大线程数

ArXiv 类别配置：
- cs.AI（人工智能）
- cs.CL（计算与语言）
- cs.CV（计算机视觉）
- cs.LG（机器学习）
- cs.MA（多智能体系统）
- cs.RO（机器人学）
- cs.MM（多媒体）
- stat.ML（统计学-机器学习）
- 等其他相关类别

## 使用方法

运行主脚本：
```bash
python get_ai_papers.py
```

### 输出文件

脚本生成两种格式的 Markdown 文件：

1. 表格格式（data/YYYY-MM/YYYY-MM-DD.md）：
   - 基本论文信息
   - 按研究方向分类
   - 简洁的表格形式

2. 详细格式（local/YYYY-MM/YYYY-MM-DD.md）：
   - 完整论文详情
   - AI 生成的分析
   - 核心贡献
   - 代码链接

## 研究类别

当前支持的研究方向：

### AI/算法主类别：
1. **AI Agents & Reasoning（智能体与推理）**
   - 单代理规划与工具使用
   - 多代理协作系统
   - 长链推理与思考链
   - 上下文工程
   - Agent 评估与基准
   - Agentic Workflow 与自动化

2. **LLM Architecture & Optimization（大模型架构与优化）**
   - 模型架构创新
   - 高效注意力机制
   - 参数高效微调
   - 模型压缩与量化
   - 推理加速
   - 模型评估与基准

3. **Multimodal & Vision-Language（多模态与视觉-语言）**
   - 视觉-语言模型
   - 多模态对齐
   - 图像/视频理解
   - 跨模态生成
   - 多模态推理

4. **Training & Optimization（训练与优化）**
   - 训练策略
   - 优化算法
   - 数据增强
   - 课程学习
   - 分布式训练

5. **Evaluation & Benchmarks（评估与基准）**
   - 模型评估方法
   - 新基准数据集
   - 性能分析
   - 鲁棒性评估

6. **Data & Knowledge（数据与知识）**
   - 数据集构建
   - 知识图谱
   - 数据质量
   - 知识增强

### 计算机视觉（CV）类别：
1. 视觉表征与基础模型
2. 视觉识别与理解
3. 生成式视觉模型
4. 三维视觉与几何推理
5. 时序视觉分析
6. 自监督与表征学习
7. 计算效率与模型优化
8. 鲁棒性与可靠性
9. 医学与生物成像
10. 视觉-语言与多模态

## 自动化部署

使用 crontab 设置每日自动运行：
```bash
# 编辑 crontab
crontab -e

# 添加以下行（每天早上9点运行）
0 9 * * * cd /path/to/ArXiv_AI_Papers_Daily/scripts && python get_ai_papers.py
```

## 错误处理

脚本包含以下错误处理机制：
- ArXiv API 速率限制处理
- 网络错误恢复
- 并行处理错误处理
- 文件系统错误处理
- LLM API 调用错误处理

## 安全注意事项

- 安全存储 API 密钥
- 监控 API 使用情况
- 保持依赖项更新
- 使用虚拟环境

## 技术细节

### LLM 分类
- 使用豆包（Doubao）大模型进行论文分类
- 支持主类别和子类别的两级分类
- 基于论文标题和摘要进行智能分类
- 返回分类置信度

### 文本预处理
- 使用 NLTK 进行文本预处理
- 支持词干提取和词形还原
- 停用词过滤
- 分词处理

### 并行处理
- 使用 ThreadPoolExecutor 进行并行处理
- 可配置的最大线程数
- 进度条显示（使用 tqdm）

## 未来改进

- 增强错误恢复能力
- 增加研究类别
- 改进代码链接检测
- 交互式网页界面
- 多语言支持
- 高级论文筛选
- 支持更多 ArXiv 类别
- 优化 LLM 分类准确性
