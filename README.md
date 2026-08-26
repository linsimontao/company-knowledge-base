# 企业知识库智能助手 —— 阶段一 MVP (第1周)

> 🏢 **项目定位**：基于真实企业制度场景构建的企业级智能知识库 RAG（Retrieval-Augmented Generation）系统。包含 15 篇全真结构化企业制度文档、语义感知的 Markdown 分块引擎、Chroma 向量数据库索引与精准检索，以及防幻觉的 LLM 答案生成。

---

## 🌟 核心功能特性

1. **15 篇高质量虚构企业制度文档**：
   - 包含员工考勤与请假、差旅报销、IT权限与VPN、入职转正、远程混合办公、设备资产领用、数据安全合规、薪酬补充医疗、绩效晋升、学习基金、会议室公约、团队建设、离职交接、知识产权与竞业协议、综合FAQ等。
2. **Markdown 标题感知语义切块（Semantic Chunking）**：
   - 自动解析 Markdown 标题层级（`#` ~ `####`），为每个分块附加完整标题路径（如 `[请假制度 > 年假细则]`），彻底解决分块切断导致的上下文丢失问题。
3. **ChromaDB 向量持久化存储与混合适配**：
   - 支持本地零配置轻量向量化（Chroma Default ONNX）、OpenAI (`text-embedding-3-small`) 及 Gemini API。
4. **防幻觉与可溯源 LLM 生成**：
   - 严格 Prompt 规则约束：基于事实回答，未知问题诚实提示并引导联系 HR/IT，所有回答在末尾标注清晰的【参考出处】。
5. **开箱即用的 CLI 交互终端与自动化测试集**：
   - 提供漂亮的 Rich 终端 UI、批量测试基准脚本及完整 Pytest 测试套件。

---

## 📐 Chunk Size 切块策略深度解析

> **核心产出物说明**：为什么这么定 Chunk Size 与 Overlap？

在企业制度类知识库中，文本切块（Chunking）的质量直接决定了向量检索的召回率与 LLM 回答的准确性：

### 1. 为什么选择 `chunk_size = 500`（中文字符）？
- **条款独立性（Clause Integrity）**：
  企业政策通常以条款或小节为单位（例如一条具体的"婚假天数与申请条件"约 150~300 字，一张"差旅住宿与餐补标准表"约 300~450 字）。500 字符刚好能够**完整容纳一个完整的制度条款或二级小节**。
- **防止语义稀释（Semantic Dilution）**：
  若 Chunk 设得太大（如 > 1500 字），一个分块中会混杂"病假"、"婚假"、"产假"多个制度，计算出的向量会被平均化（Diluted），导致检索精准度下降。
- **防止碎片化（Over-segmentation）**：
  若 Chunk 设得太小（如 < 150 字），会把主谓宾和适用条件切碎（例如只切到"可报销 80%"，却丢失了前提是"社保范围外自费药"），造成断章取义。

### 2. 为什么选择 `chunk_overlap = 100`（约 20% 重叠）？
- **边界平滑与跨段衔接**：
  当某个制度条款较长必须跨块拆分时，100 字符的滑动重叠（Overlap）能够确保切分边界处不会丢失因果关系与定语条件（例如前一段末尾的"单笔金额超过 5,000 元的支出"与后一段开头的"须由财务总监审批"在重叠区保留）。

### 3. 标题层级注入（Header Breadcrumb Injection）
每个 Chunk 在入库前均被强制注入面包屑上下文：
`[01_员工考勤与请假管理制度.md > 第二章 假期类型与计算标准 > 2.1 法定带薪年休假（年假）]`
使原本只有"工龄满10年享受10天"的孤立片段携带了完整的业务语义背景。

---

## 📁 目录结构

```
company-knowledge-base/
├── data/
│   ├── documents/                     # 15 篇企业 Markdown 制度文档
│   │   ├── 01_员工考勤与请假管理制度.md
│   │   ├── 02_企业差旅与费用报销管理办法.md
│   │   ├── 03_IT基础权限与系统接入指引.md
│   │   ├── 04_公司入职与试用期转正指南.md
│   │   ├── 05_远程办公与混合工作制度.md
│   │   ├── 06_办公资产与设备领用报修规范.md
│   │   ├── 07_企业信息安全与数据合规守则.md
│   │   ├── 08_员工薪酬福利与补充商业保险说明.md
│   │   ├── 09_员工绩效考核与职级晋升管理办法.md
│   │   ├── 10_培训发展与学习基金申领制度.md
│   │   ├── 11_办公区行为规范与会议室使用公约.md
│   │   ├── 12_员工关怀与文体活动经费指引.md
│   │   ├── 13_离职与资产交接办理流程.md
│   │   ├── 14_知识产权与竞业限制协议解读.md
│   │   └── 15_员工综合常见问题解答(FAQ).md
│   └── chroma_db/                     # Chroma 向量数据库本地持久化目录
├── src/                               # 核心 RAG 框架代码
│   ├── __init__.py
│   ├── config.py                      # 统一配置中心 (Pydantic + dotenv)
│   ├── chunker.py                     # Markdown 标题感知语义切块器
│   ├── embeddings.py                  # 多后端向量化适配器 (Chroma/OpenAI/Gemini/Mock)
│   ├── vector_store.py                # ChromaDB 向量集合管理与索引
│   ├── retriever.py                   # 相似度打分与检索器
│   ├── generator.py                   # 防幻觉 Prompt 构造与 LLM 生成
│   └── rag_pipeline.py                # 端到端 RAG 编排引擎
├── scripts/
│   ├── build_index.py                 # 构建/重建知识库索引 CLI
│   └── query_rag.py                   # 智能问答 CLI (支持交互式与自动化测试)
├── tests/
│   └── test_rag_pipeline.py           # 自动化测试用例
├── .env.example                       # 环境变量示例
├── requirements.txt                   # 项目依赖
└── README.md                          # 本项目说明
```

---

## 🚀 快速上手与运行指引

### 1. 激活虚拟环境与安装依赖
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. (可选) 配置大模型 API Key
如果需要接入 OpenAI、DeepSeek 或 Gemini 等线上大模型，可复制 `.env.example` 为 `.env` 并填写 API Key：
```bash
cp .env.example .env
# 编辑 .env 填入 OPENAI_API_KEY 或 GEMINI_API_KEY
```
*注：系统默认无需任何 API Key，可直接基于内置本地 ONNX 向量模型与离线提取引擎无缝体验全流程。*

### 3. 一键构建向量索引
```bash
python scripts/build_index.py
```

### 4. 运行问答与测试用例

**方式 A：运行标准基准测试集（包含"年假"、"报销"、"VPN"等核心问题）**
```bash
python scripts/query_rag.py --test-all
```

**方式 B：单问题直接查询**
```bash
python scripts/query_rag.py --query "年假一年有多少天"
python scripts/query_rag.py --query "报销需要哪些材料"
python scripts/query_rag.py --query "怎么申请VPN权限"
```

**方式 C：进入交互式 CLI 问答终端**
```bash
python scripts/query_rag.py
```

### 5. 运行自动化测试套件
```bash
pytest tests/ -v
```

