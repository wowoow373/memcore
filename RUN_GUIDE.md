# mem0 本地开发与测试指南

本文档说明如何在该项目环境下运行代码和测试。

---

## 一、环境准备

### 1. Conda 环境

所有代码**必须**运行在 `mem0` conda 环境中：

```bash
# 激活环境
conda activate mem0

# 验证 Python 版本
python --version
# 预期输出：Python 3.11.x
```

如果环境不存在，可创建：

```bash
conda create -n mem0 python=3.11 -y
conda activate mem0
pip install -e ".[test,graph,vector_stores,llms,extras]"
```

---

## 二、数据库服务启动

项目依赖 PostgreSQL (带 pgvector 扩展) 和 Neo4j，通过 Docker Compose 启动。

### 启动容器

```bash
cd server/
docker compose up -d
```

> Docker Compose 文件位置：`server/docker-compose.yaml`

### 验证服务状态

```bash
docker compose ps
```

两个服务都应处于 `healthy` 状态。

### 端口映射

| 服务      | 容器内部端口 | 映射到宿主机端口 | 说明             |
|-----------|-------------|----------------|------------------|
| PostgreSQL| 5432        | 8432           | 向量数据库       |
| Neo4j     | 7474        | 8474           | Neo4j HTTP 浏览器|
| Neo4j     | 7687        | 8687           | Neo4j Bolt 协议  |

- PostgreSQL 连接：`postgresql://postgres:postgres@localhost:8432/postgres`
- Neo4j 浏览器：http://localhost:8474
- Neo4j Bolt：`bolt://localhost:8687`

---

## 三、环境变量

项目配置通过 `server/.env` 文件管理。**运行代码前，先加载该文件中的环境变量。**

```bash
# 在项目根目录下执行
export $(grep -v '^#' server/.env | xargs)
```

> **注意**：`server/.env` 中已包含完整的配置，**不要**将真实密钥写入任何文档或脚本。如有需要单独设置某个变量，请使用占位符：
>
> ```bash
> export OPENAI_llm_API_KEY="<YOUR_API_KEY>"
> ```

---

## 四、运行代码

在确保 conda 环境已激活、Docker 服务已启动、环境变量已加载后：

### 运行 Python 脚本

```bash
conda activate mem0
export $(grep -v '^#' server/.env | xargs)

# 示例：运行测试脚本或调用 Memory API
python your_script.py
```

### 启动 REST API Server

```bash
conda activate mem0
cd server/
export $(grep -v '^#' .env | xargs)
uvicorn main:app --reload --port 8888
```

---

## 五、运行测试

### 全部测试

```bash
conda activate mem0
export $(grep -v '^#' server/.env | xargs)
pytest tests/
```

### 单个测试文件

```bash
pytest tests/<path_to_test_file>.py
```

### 指定测试用例

```bash
pytest tests/<path_to_test_file>.py -k <test_case_name>
```

> 具体测试文件和用例请阅读 `tests/` 目录下的相关代码。

---

## 六、完整工作流示例

```bash
# 1. 进入项目目录
cd /home/wowoow/open-source/mem0-main

# 2. 激活 conda 环境
conda activate mem0

# 3. 启动数据库服务
cd server/
docker compose up -d
cd ..

# 4. 加载环境变量
export $(grep -v '^#' server/.env | xargs)

# 5. 运行测试
pytest tests/memory/test_search_engine.py -v

# 6. 或直接运行代码
python your_script.py
```

---

## 七、常见问题

### Docker 服务未启动

如果测试报错连接失败，先检查容器状态：

```bash
cd server/
docker compose ps
docker compose logs postgres
docker compose logs neo4j
```

### 环境变量未生效

Python 进程不会自动读取 `.env` 文件，必须通过 `export` 或 `source` 加载。

### 端口冲突

如果本地 8432 / 8474 / 8687 端口被占用，修改 `server/docker-compose.yaml` 中的端口映射后重启容器。

---

## 八、停止服务

```bash
cd server/
docker compose down
```

如需清空数据卷（谨慎操作）：

```bash
docker compose down -v
```
