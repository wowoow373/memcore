FROM python:3.12  
# 基础镜像：使用 Python 3.12 官方镜像，确保环境具有最新的解释器特性

WORKDIR /app  
# 设置工作目录：后续所有指令（COPY, RUN等）都会在容器内的 /app 目录下执行

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -  
# 安装 Poetry：官方安装脚本，用于后续管理项目依赖
ENV PATH="/root/.local/bin:$PATH"  
# 配置环境变量：将 Poetry 的安装路径添加到 PATH，确保可以直接调用 poetry 命令

# Copy requirements first for better caching
COPY server/requirements.txt .  
# 缓存优化：先只复制依赖描述文件，防止代码变动导致重复安装依赖
RUN pip install -r requirements.txt  
# 安装基础依赖：使用 pip 安装 server 运行所需的核心库

# Install mem0 in editable mode using Poetry
WORKDIR /app/packages  
# 切换目录：进入包管理目录准备安装本地 library
COPY pyproject.toml .  
# 复制项目配置文件：包含 mem0 的元数据和依赖定义
COPY poetry.lock .     
# 复制锁定文件：确保安装的版本与开发环境完全一致
COPY README.md .       
# 复制说明文档：通常 pyproject.toml 会引用此文件作为包描述
COPY mem0 ./mem0       
# 复制 mem0 库的源代码
RUN pip install -e .[graph]  
# 可编辑模式安装：-e 表示以 Link 形式安装，配合 [graph] 额外安装图数据库相关的依赖（如 neo4j 驱动）

# Return to app directory and copy server code
WORKDIR /app  
# 切回主程序目录
COPY server .  
# 复制服务器应用代码：最后一步复制业务逻辑，最大化利用之前的镜像层缓存

# 启动命令：使用 uvicorn 运行 FastAPI 应用
# --host 0.0.0.0：允许外部访问
# --reload：监测到代码变动时自动重启服务，适合开发环境
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]