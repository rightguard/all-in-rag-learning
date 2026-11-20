import os
import re
from collections import Counter
from datetime import datetime

import torch
import yaml
from dotenv import load_dotenv
from tqdm.auto import tqdm
from unstructured.partition.auto import partition

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 路径配置（WSL Linux 格式，/mnt/e 对应 Windows E 盘）
PROJECT_ROOT = "/mnt/e/all-in-rag"
ENV_PATH = f"{PROJECT_ROOT}/learn/.env"
CONFIG_PATH = f"{PROJECT_ROOT}/learn/config.yaml"
DATA_PATH = f"{PROJECT_ROOT}/data/C2/pdf/rag.pdf"
EMBEDDING_MODEL_PATH = f"{PROJECT_ROOT}/model/bge_small_zh_v1.5"
OUTPUT_PATH = f"{PROJECT_ROOT}/learn/char2/01_unstructured_output.md"
EXTRACTED_PATH = f"{PROJECT_ROOT}/learn/char2/01_unstructured_extracted.md"

STRATEGY_NAME = "Unstructured 解析 PDF + 递归字符分块"
QUESTION = "这篇文档主要讲了什么？RAG（检索增强生成）的基本流程包括哪些步骤？"

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

# 1. Unstructured 解析 PDF
print(f"[Parser] 正在解析 PDF: {DATA_PATH}")
elements = partition(filename=DATA_PATH, content_type="application/pdf")
types = Counter(e.category for e in elements)
print(f"[Parser] 解析完成: {len(elements)} 个元素 | 类型统计: {dict(types)}")

# 元素转 Document，category / 来源写入 metadata 便于溯源
docs = []
with tqdm(total=len(elements), desc="解析元素转文档", unit="元素") as pbar:
    for element in elements:
        text = getattr(element, "text", str(element)).strip()
        if text:
            docs.append(Document(page_content=text, metadata={
                "category": getattr(element, "category", "unknown"),
                "source": DATA_PATH,
            }))
        pbar.update(1)
print(f"[Parser] 有效文本元素: {len(docs)} 个")

# 把提取内容转成 Markdown：过滤网页打印 PDF 的噪声，平铺为段落
# unstructured 对网页型 PDF 的 Title 分类不可靠，故不再渲染标题层级
nav_words = {
    "网页", "新闻", "贴吧", "知道", "网盘", "图片", "视频", "地图", "文库",
    "资讯", "采购", "百科", "首页", "登录", "注册", "收藏", "播报", "帮助",
    "上传视频", "秒懂百科", "特色百科", "知识专题", "加入百科", "百科团队",
    "权威合作", "全站搜索", "进入词条", "讨论", "锁定", "查",
}


def is_noise(text, category):
    if category in ("Footer", "PageBreak"):
        return True
    if text in nav_words:
        return True
    if len(text) <= 3 and category == "Title":
        return True
    if re.fullmatch(r"[\d\s:.\-]+", text):
        return True
    return False


extracted_lines = [f"# PDF 提取内容：{DATA_PATH}", ""]
with tqdm(total=len(elements), desc="提取内容转 Markdown", unit="元素") as pbar:
    for element in elements:
        text = getattr(element, "text", str(element)).strip()
        category = getattr(element, "category", "")
        if text and not is_noise(text, category):
            extracted_lines += [text, ""]
        pbar.update(1)

with open(EXTRACTED_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(extracted_lines))
print(f"[Output] 提取内容已写入: {EXTRACTED_PATH}")

# 2. 文本分块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunks = text_splitter.split_documents(docs)
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
    f"- 解析元素数：{len(elements)}",
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
