import os
from datetime import datetime

import torch
import yaml
from dotenv import load_dotenv
from tqdm.auto import tqdm

from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# 新版在 langchain_text_splitters，旧版在 langchain_experimental
try:
    from langchain_text_splitters import SemanticChunker
except ImportError:
    from langchain_experimental.text_splitter import SemanticChunker

# 路径配置（WSL Linux 格式，/mnt/e 对应 Windows E 盘）
PROJECT_ROOT = "/mnt/e/all-in-rag"
ENV_PATH = f"{PROJECT_ROOT}/learn/.env"
CONFIG_PATH = f"{PROJECT_ROOT}/learn/config.yaml"
DATA_PATH = f"{PROJECT_ROOT}/data/C2/txt/蜂医.txt"
EMBEDDING_MODEL_PATH = f"{PROJECT_ROOT}/model/bge_small_zh_v1.5"
OUTPUT_PATH = f"{PROJECT_ROOT}/learn/char2/04_semantic_output.md"

STRATEGY_NAME = "语义分块（SemanticChunker / percentile）"
QUESTION = "请介绍一下蜂医这个角色，他有哪些技能和特点？"

load_dotenv(ENV_PATH, override=True)
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

embedding_config = config.get("embedding", {})
REQUESTED_DEVICE = str(embedding_config.get("device", "auto")).strip().lower()
EMBEDDING_BATCH_SIZE = int(embedding_config.get("batch_size", 32))
NORMALIZE_EMBEDDINGS = bool(embedding_config.get("normalize_embeddings", True))


# 优先 CUDA，不可用时回退 CPU
def resolve_device(requested: str) -> str:
    cuda_available = torch.cuda.is_available()
    if requested.startswith("cuda"):
        if cuda_available:
            return requested if ":" in requested else "cuda:0"
        print("[警告] 配置指定 CUDA，但 torch.cuda.is_available() 为 False，已自动回退到 CPU。")
        return "cpu"
    if requested == "auto":
        return "cuda:0" if cuda_available else "cpu"
    return "cpu"


device = resolve_device(REQUESTED_DEVICE)
if device.startswith("cuda"):
    gpu_index = int(device.split(":")[1]) if ":" in device else 0
    total_vram_gb = torch.cuda.get_device_properties(gpu_index).total_memory / 1024 ** 3
    print(f"[GPU] 使用设备: {device} | 显卡: {torch.cuda.get_device_name(gpu_index)} "
          f"| 显存: {total_vram_gb:.1f} GiB | 可见 GPU 数: {torch.cuda.device_count()}")
else:
    print("[GPU] 使用设备: cpu（未检测到可用 CUDA）")


class BatchedBgeEmbeddings(HuggingFaceEmbeddings):
    """固定 batch 分批向量化，并用 tqdm 展示进度。"""

    def embed_documents(self, texts):
        vectors = []
        with tqdm(total=len(texts), desc="Embedding 向量化", unit="段") as pbar:
            for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
                batch = texts[start:start + EMBEDDING_BATCH_SIZE]
                vectors.extend(super().embed_documents(batch))
                pbar.update(len(batch))
        return vectors


embeddings = BatchedBgeEmbeddings(
    model_name=EMBEDDING_MODEL_PATH,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": NORMALIZE_EMBEDDINGS},
)
print(f"[Embedding] 模型: {EMBEDDING_MODEL_PATH} | batch_size: {EMBEDDING_BATCH_SIZE} "
      f"| normalize: {NORMALIZE_EMBEDDINGS}")

# 1. 文档加载
loader = TextLoader(DATA_PATH, encoding="utf-8")
documents = loader.load()
print(f"[Loader] 成功加载 {len(documents)} 个文档")

# 2. 语义分块（内部同样调用上面的 embeddings，带 tqdm 进度条）
text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
chunks = text_splitter.split_documents(documents)
print(f"[Splitter] 文本被切分为 {len(chunks)} 个块")

# 3. 构建向量存储
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)

# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:"""
                                          )

# 4. 配置大语言模型
model_config = config["model"]
llm = ChatOpenAI(
    model=model_config["name"],
    temperature=model_config["temperature"],
    top_p=model_config["top_p"],
    max_tokens=model_config["max_tokens"],
    api_key=os.getenv("QWEN_API_KEY"),
    base_url=os.getenv("QWEN_API_BASE_URL"),
)

# 5. 检索 + 调用大模型
retrieved_docs = vectorstore.similarity_search(QUESTION, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
with tqdm(total=1, desc="LLM 生成回答", unit="次") as pbar:
    answer = llm.invoke(prompt.format(question=QUESTION, context=docs_content))
    pbar.update(1)

answer_text = answer.content if hasattr(answer, "content") else str(answer)

# 6. 输出 Markdown
md_lines = [
    "# RAG 问答结果",
    "",
    f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"- 大模型：{model_config['name']}",
    f"- 嵌入模型：{EMBEDDING_MODEL_PATH}",
    f"- 运行设备：{device}",
    f"- 分块策略：{STRATEGY_NAME}",
    f"- 数据文件：{DATA_PATH}",
    f"- 切分块数：{len(chunks)}",
    f"- 检索片段数：{len(retrieved_docs)}",
    "",
    "## 问题",
    "",
    QUESTION,
    "",
    "## 检索到的上下文",
    "",
]
for idx, doc in enumerate(retrieved_docs, 1):
    md_lines += [f"### 片段 {idx}", "", doc.page_content, ""]
md_lines += ["## 回答", "", answer_text, ""]

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print(f"[Output] 结果已写入: {OUTPUT_PATH}")
print(answer_text)
