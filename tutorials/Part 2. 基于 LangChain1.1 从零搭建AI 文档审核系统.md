<!-- source: Part 2. 基于 LangChain1.1 从零搭建AI 文档审核系统.ipynb -->

# <center>Part 2. 基于 LangChain1.1 从零搭建AI 文档审核系统</center>

<div align=center><img src="Part 2. 基于 LangChain1.1 从零搭建AI 文档审核系统.assets/remote/202512171421744-c29a262eb9.png" width=80%></div>

&emsp;&emsp;借助大语言模型的能力，我们可以构建一个<font color=red>智能文档审核系统</font>，自动识别文档中的语法错误、不当用语、敏感词汇以及一些定制化的审核规则，大幅提升文档审核效率。本小节课程，我们就从零开始，使用 `LangChain1.1` 框架和 `DeepSeek-v3.2` 大模型来进行文档审核Agent系统的开发。

&emsp;&emsp;具体来说，我们将实现以下完整功能的开发及运行：

<style>
.center
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

<p align="center"><font face="黑体" size=4>核心功能</font></p>
<div class="center">

| 步骤 | 说明 | 产出 |
|------|------|------|
| **输入** | 真实 PDF 文档 | PDF 文件 |
| **解析** | 使用 `MinerU` 解析 | 结构化文本 + 坐标 |
| **审核** | 用 `LangChain 1.1` 调用大模型，按"预设规则 + 自定义规则"审核 | 结构化问题列表 |
| **输出** | 将审核结果整理成"问题列表"（表格/JSON） | 可用于前端高亮与整改 |

</div>

&emsp;&emsp;具体应用到的核心技术栈如下：

<style>
.center
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

<p align="center"><font face="黑体" size=4>核心技术栈</font></p>
<div class="center">

| 技术 | 作用 | 说明 |
|------|------|------|
| `LangChain1.1` | LLM 应用开发框架 | 简化大模型调用，提供统一接口 |
| `DeepSeek-v3.2` | 大语言模型 | 国产高性能 LLM，性价比高 |
| `MinerU` | PDF 解析服务 | 将 PDF 转换为结构化文本 |
| `Pydantic` | 数据验证 | 定义结构化输出格式 |

</div>

## 1. 环境准备与依赖安装

### 1.1 安装必要的依赖库

&emsp;&emsp;首先，我们需要安装本课程所需的 Python 库。在当前的 Jupyter Notebook 中运行以下命令：

```python
# 安装核心依赖
%pip install langchain langchain-openai pydantic httpx python-dotenv -q
```

&emsp;&emsp;这些库的作用分别如下：

- `langchain`：LLM 应用开发核心框架
- `langchain-openai`：LangChain 的 OpenAI 兼容接口（DeepSeek 使用 OpenAI 兼容协议）
- `pydantic`：数据验证和结构化输出
- `httpx`：异步 HTTP 客户端，用于调用 MinerU API
- `python-dotenv`：环境变量管理

### 1.2 准备 API 密钥

&emsp;&emsp;在使用 DeepSeek 和 MinerU 之前，我们需要获取并配置 API 密钥。

1. **DeepSeek API Key**：访问 [DeepSeek 开放平台](https://platform.deepseek.com/)，注册账号后获取 API Key；
2. **MinerU API Key**：访问 [MinerU 官网](https://mineru.net/)，需要先进行API申请，然后创建 Token 即可。

<div align=center><img src="Part 2. 基于 LangChain1.1 从零搭建AI 文档审核系统.assets/remote/202512171429752-2919f31132.png" width=80%></div>

&emsp;&emsp;当然，除了`MinerU`，我们还可以使用`PaddleOCR-VL`、`DeepSeek-OCR`等其他解析方案，以上三种解析方案均有本地部署版本，大家可以找到助教老师领取资料：

<div align=center><img src="Part 2. 基于 LangChain1.1 从零搭建AI 文档审核系统.assets/remote/202511071654511-6127cdbcf4.png" width=70%></div>

<div align=center><img src="Part 2. 基于 LangChain1.1 从零搭建AI 文档审核系统.assets/remote/202512171342119-fb14d0b41e.png" width=30%></div>

&emsp;&emsp;`MinerU`是一个非常典型的基于管道的解决方案 (Pipeline-based solution) ，并且是一个开源文档解析项目，一共四个核心组件，通过`Pipeline`的设计无缝衔接，实现比较高效、准确的文档解析。如下图所示：（下图来源官方论文：https://arxiv.org/pdf/2409.18839v1）

<div align=center><img src="Part 2. 基于 LangChain1.1 从零搭建AI 文档审核系统.assets/remote/202503271440433-e5b3b72767.png" width=70%></div>

&emsp;&emsp;`MinerU`的主要工作流程分为以下几个阶段：

1. **输入**：接收`PDF` 格式文本，可以是简单的纯文本，也可以是包含双列文本、公式、表格或图等多模态`PDF`文件;
2. **文档预处理（Document Preprocessing）**：检查语言、页面大小、文件是否被扫描以及加密状态；
3. **内容解析（Content Parsing）**：
    - 局分析：区分文本、表格和图像。
    - 公式检​​测和识别：识别公式类型（内联、显示或忽略）并将其转换为 `LaTeX` 格式。
    - 表格识别：以 `HTML/LaTeX` 格式输出表格。
    - OCR：对扫描的 `PDF` 执行文本识别。
4. **内容后处理（Content Post-processing）**：修复文档解析后可能出现的问题。比如解决文本、图像、表格和公式块之间的重叠，并根据人类阅读模式重新排序内容，确保最终输出遵循自然的阅读顺序。
5. **格式转换（Format Conversion）**：以 `Markdown` 或 `JSON` 格式生成输出。
6. **输出（Output）**：高质量、结构良好的解析文档。

&emsp;&emsp;目前在`Github` 上，`MinerU` 的`Star` 数为`48.2K`，`Fork` 数为`4K`，拥有非常良好的社区支持和活跃的贡献者，且一直处于`active development`状态。

<div align=center><img src="Part 2. 基于 LangChain1.1 从零搭建AI 文档审核系统.assets/remote/202511071014588-ce347c5709.png" width=80%></div>

&emsp;&emsp;`MinerU` 提供了在线`Demo` 页面，我们可以直接线进行测试。试用地址：https://opendatalab.com/OpenSourceTools/Extractor/PDF/

<div align=center><img src="Part 2. 基于 LangChain1.1 从零搭建AI 文档审核系统.assets/remote/202503271502432-0e4f5b0a81.png" width=70%></div>

&emsp;&emsp;同时，`MinerU` 项目于`2024年07月05日`首次开源，底层主要是集成 `PDF-Extract-Kit` 开源项目做`PDF`的内容提取，`PDF-Extract-Kit`同样是一个开源项目：https://github.com/opendatalab/PDF-Extract-Kit

<div align=center><img src="Part 2. 基于 LangChain1.1 从零搭建AI 文档审核系统.assets/remote/202503251117693-4e3b05dcf9.png" width=70%></div>

&emsp;&emsp;`PDF-Extract-Kit`这个项目主要针对的是`PDF`文档的内容提取，通过集成众多`SOTA`模型对`PDF`文件实现高质量的内容提取，其中应用到的模型主要包括：

<style>
.center
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

<p align="center"><font face="黑体" size=4>PDF-Extract-Kit 应用的模型类型</font></p>
<div class="center">


| 模型类型       | 模型名称         | GitHub 链接                                                                 | 模型下载链接                                                       | 任务描述                                           |
|----------------|------------------|---------------------------------------------------------------------------|-------------------------------------------------------------------|----------------------------------------------------|
| 布局检测模型   | LayoutLMv3 / DocLayout-YOLO | [GitHub](https://github.com/microsoft/unilm/tree/master/layoutlmv3) / [GitHub](https://github.com/opendatalab/DocLayout-YOLO) | [模型下载](https://huggingface.co/microsoft/layoutlmv3-base-chinese) / [模型下载](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench/tree/main) | 定位文档中不同元素位置：包含图像、表格、文本、标题、公式等 |
| 公式检测模型   | YOLO             | [GitHub](https://github.com/ultralytics/ultralytics)                     | [模型下载](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt) | 定位文档中公式位置：包含行内公式和行间公式         |
| 公式识别模型   | UniMERNet        | [GitHub](https://github.com/opendatalab/UniMERNet)                       | [模型下载](https://huggingface.co/wanderkid/unimernet_base)      | 识别公式图像为latex源码                             |
| 表格识别模型   | StructEqTable    | [GitHub](https://github.com/Alpha-Innovator/StructEqTable-Deploy?tab=readme-ov-file) | [模型下载](https://huggingface.co/U4R/StructTable-InternVL2-1B/tree/main) | 识别表格图像为对应源码（Latex/HTML/Markdown）     |
| OCR模型        | PaddleOCR        | [GitHub](https://github.com/PaddlePaddle/PaddleOCR)                      | [模型下载](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/text_detection.html)                                                                  | 提取图像中的文本内容（包括定位和识别）             |
</div>

&emsp;&emsp;`MinerU`与`PDF-Extract-Kit`的关系是：`MinerU` 结合`PDF-Extract-Kit`输出的高质量预测结果，进行了专门的工程优化，使得文档内容提取更加便捷高效，处理底层原理的优化细节外，主要提升点在以下几点：

- 加入了自研的`doclayout_yolo(2501)`模型做布局检测，在相近解析效果情况下比原方案提速10倍以上，可以通过配置文件与 `layoutlmv3` 自由切换使用；
- 加入了自研的`unimernet(2501)` 模型做公式识别，针对真实场景下多样性公式识别的算法，可以对复杂长公式、手写公式、含噪声的截图公式均有不错的识别效果；
- 增加 `OCR` 的多语言支持，支持 `84` 种语言的检测与识别，支持列表：https://paddlepaddle.github.io/PaddleOCR/latest/ppocr/blog/multi_languages.html#5
- 重构排序模块代码，使用 `layoutreader` 进行阅读顺序排序，确保在各种排版下都能实现极高准确率 : https://github.com/ppaanngggg/layoutreader
- 表格识别功能接入了`StructTable-InternVL2-1B`模型，大幅提升表格识别效果，模型下载地址:https://huggingface.co/U4R/StructTable-InternVL2-1B

&emsp;&emsp;这里为了帮助大家降低接入的难度，我们就直接使用在线的MinerU API。在获取密钥后，可以通过以下方式配置：

```python
# 方式一：直接在代码中配置（仅用于学习，生产环境不推荐）
# DEEPSEEK_API_KEY = "your-deepseek-api-key-here"
# MINERU_API_KEY = "your-mineru-api-key-here"

# 方式二：使用环境变量（推荐）
import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MINERU_API_KEY = os.getenv("MINERU_API_KEY")

# 验证配置
print(f"DeepSeek API Key 已配置: {DEEPSEEK_API_KEY[:10]}")
print(f"MinerU API Key 已配置: {MINERU_API_KEY[:10]}")
```

&emsp;&emsp;如果可以正常显示`DeepSeek API Key` 和 `MinerU API Key`的前10位，说明环境变量加载成功。

## 2. 使用 MinerU 解析 PDF

<div align=center><img src="Part 2. 基于 LangChain1.1 从零搭建AI 文档审核系统.assets/remote/202512171522023-39682fe702.png" width=80%></div>

&emsp;&emsp;MinerU 在线API的工作流程如下：

```json
    1. 请求上传 URL  ───▶  2. 上传 PDF  ───▶  3. 轮询解析状态  ───▶  4. 下载解析结果
```

&emsp;&emsp;大家可以在MinerU的官网看到详细的API接口文档：

<div align=center><img src="Part 2. 基于 LangChain1.1 从零搭建AI 文档审核系统.assets/remote/202512171440380-4d54def014.png" width=70%></div>


&emsp;&emsp;下面我们来实现一个MinerU 客户端：

```python
import httpx
import asyncio
import time
import zipfile
import io
import json
from pathlib import Path

async def parse_pdf_with_mineru(pdf_path: str, api_key: str) -> list[dict]:
    """
    使用 MinerU API 解析 PDF 文档。

    参数:
        pdf_path: PDF 文件路径
        api_key: MinerU API 密钥

    返回:
        解析后的段落列表
    """
    base_url = "https://mineru.net"
    file_path = Path(pdf_path)
    file_name = file_path.name

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=300) as client:
        # 步骤1：请求预签名上传 URL
        print("步骤1：请求上传 URL...")
        resp = await client.post(
            f"{base_url}/api/v4/file-urls/batch",
            headers=headers,
            json={
                "files": [{"name": file_name, "data_id": file_name}],
                "model_version": "vlm",  # 使用视觉语言模型
            }
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        batch_id = data["batch_id"]
        upload_url = data["file_urls"][0]
        print(f"获取到 batch_id: {batch_id}")

        # 步骤2：上传 PDF 文件
        print("步骤2：上传 PDF 文件...")
        with open(pdf_path, "rb") as f:
            await client.put(upload_url, content=f.read())
        print("上传完成")

        # 步骤3：轮询等待解析完成
        print("步骤3：等待解析完成...")
        max_wait = 300  # 最多等待5分钟
        start_time = time.time()

        while True:
            resp = await client.get(
                f"{base_url}/api/v4/extract-results/batch/{batch_id}",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            result = resp.json()["data"]["extract_result"][0]

            if result["state"] == "done":
                full_zip_url = result["full_zip_url"]
                print("解析完成")
                break
            elif result["state"] == "failed":
                raise RuntimeError(f"解析失败: {result.get('err_msg')}")

            if time.time() - start_time > max_wait:
                raise TimeoutError("解析超时")

            await asyncio.sleep(2)  # 每2秒检查一次

        # 步骤4：下载并解析结果
        print("步骤4：下载解析结果...")
        resp = await client.get(full_zip_url)
        zip_bytes = resp.content

        # 解压并提取 JSON
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            json_files = [n for n in zf.namelist() if n.endswith(".json")]
            if not json_files:
                raise RuntimeError("未找到解析结果")

            with zf.open(json_files[0]) as f:
                content = json.loads(f.read().decode("utf-8"))

        print("下载完成")

        # 步骤5：提取段落文本
        paragraphs = extract_paragraphs(content)
        print(f"共提取了 {len(paragraphs)} 个段落")

        return paragraphs
```

&emsp;&emsp;上面的代码实现了与 MinerU API 的完整交互流程。接下来，我们需要一个辅助函数来从 MinerU 返回的 JSON 中提取段落文本：

```python
def extract_paragraphs(content) -> list[dict]:
    """
    从 MinerU 解析结果中提取段落文本。

    MinerU 返回的 JSON 结构可能有多种格式，这个函数会尝试兼容不同格式。
    """
    paragraphs = []

    # 情况1：content 是列表（每个元素是一个文本块）
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text.strip():
                    paragraphs.append({
                        "content": text.strip(),
                        "page_num": item.get("page_idx", 0) + 1,
                        "bbox": item.get("bbox"),
                    })
        return paragraphs

    # 情况2：content 是字典，包含 pages 字段
    if isinstance(content, dict):
        pages = content.get("pages") or []
        for page in pages:
            page_num = page.get("page", 1)
            blocks = page.get("paragraphs") or page.get("blocks") or []
            for block in blocks:
                text = block.get("text") or block.get("content") or ""
                if text.strip():
                    paragraphs.append({
                        "content": text.strip(),
                        "page_num": page_num,
                        "bbox": block.get("bbox"),
                    })

    return paragraphs

# 测试用的示例数据
sample_paragraphs = [
    {"content": "本公司承诺绝对保证产品质量，必须满足所有客户需求。", "page_num": 1},
    {"content": "根据市场调研，我们的产品销量将一定达到预期目标。", "page_num": 1},
    {"content": "公司简介：我们是一家专注于人工智能领域的科技公司。", "page_num": 2},
    {"content": "团队介绍：我们拥有一只经验丰富的研发团队。", "page_num": 2},
]

print("示例段落数据：")
for i, p in enumerate(sample_paragraphs):
    print(f"  [{i}] 第{p['page_num']}页: {p['content'][:30]}...")
```

&emsp;&emsp;进行真实PDF解析测试：

```python
# 1. 本地 PDF 路径（改成你自己的）
pdf_path = "./data/LangChain v1.1 文档审核类Agent开发实战.pdf"

# 2. MinerU API Key（推荐走环境变量）
api_key = os.environ.get("MINERU_API_KEY")
if not api_key:
    raise RuntimeError("请先设置环境变量 MINERU_API_KEY")

# 3. 调用解析函数
paragraphs = await parse_pdf_with_mineru(pdf_path, api_key)

# 4. 打印结果
print("\n====== 解析结果 ======")
for i, p in enumerate(paragraphs):
    print(f"[{i}] 第{p['page_num']}页: {p['content'][:80]}")
```

&emsp;&emsp;相信大家看到这里已经理解了 PDF 解析的基本流程。接下来，让我们进入最核心的部分——使用 LangChain 调用大模型进行文档审核。

## 3. LangChain 1.1 快速入门及应用

&emsp;&emsp;`LangChain` 是一个用于开发大语言模型（LLM）应用的框架。

<div align=center><img src="Part 2. 基于 LangChain1.1 从零搭建AI 文档审核系统.assets/remote/202512171505664-7c6ad87493.png" width=80%></div>

&emsp;&emsp;它的核心价值在于：

1. **统一接口**：无论使用 OpenAI、DeepSeek、Claude 还是其他模型，调用方式都一样
2. **链式调用**：可以将多个处理步骤串联起来，形成一个完整的处理流程
3. **丰富的工具**：提供了 Prompt 模板、输出解析器、记忆组件等实用工具

&emsp;&emsp;在本课程中，我们主要使用 LangChain 的以下功能：

- **ChatOpenAI**：调用兼容 OpenAI 接口的大模型（如 DeepSeek-v3.2）
- **消息类型**：`SystemMessage`（系统提示）、`HumanMessage`（用户输入）
- **PydanticOutputParser**：将 LLM 输出解析为结构化的 Python 对象

### 3.1 初始化 DeepSeek-v3.2 模型

&emsp;&emsp;我们首先初始化一个 DeepSeek-v3.2 模型实例。DeepSeek 使用与 OpenAI 兼容的 API 接口，所以我们可以直接使用 `ChatOpenAI` 类来调用它。

```python
from langchain_openai import ChatOpenAI

def init_llm(api_key: str) -> ChatOpenAI:
    """
    初始化 DeepSeek 大语言模型。

    参数:
        api_key: DeepSeek API 密钥

    返回:
        ChatOpenAI 实例
    """
    llm = ChatOpenAI(
        model="deepseek-chat",           # DeepSeek 模型名称
        api_key=api_key,                  # API 密钥
        base_url="https://api.deepseek.com/v1",  # DeepSeek API 地址
        temperature=0.2,                  # 温度参数，越低输出越稳定
        max_tokens=4096,                  # 最大输出长度
    )
    return llm

# 初始化模型
llm = init_llm(DEEPSEEK_API_KEY)
print("大模型初始化完成")
```

&emsp;&emsp;简单测试一下模型是否正常工作：

```python
from langchain_core.messages import HumanMessage

# 发送一条简单的测试消息
response = llm.invoke([HumanMessage(content="你好，请用一句话介绍你自己。")])
print(f"模型回复：{response.content}")
```

> &emsp;**参数说明**：`temperature` 参数控制模型输出的随机性。在文档审核场景中，我们希望输出稳定一致，所以设置为较低的值（0.2）。

### 3.2 LangChain消息类型详解

&emsp;&emsp;在 LangChain 中，与大模型的对话由不同类型的「消息」组成。理解这些消息类型对于构建有效的 Prompt 非常重要：


<style>
.center
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

<p align="center"><font face="黑体" size=4>LangChain 消息类型</font></p>
<div class="center">

| 消息类型 | 作用 | 使用场景 |
|----------|------|----------|
| `SystemMessage` | 系统提示 | 定义 AI 的角色、行为规范 |
| `HumanMessage` | 用户消息 | 用户的输入内容 |
| `AIMessage` | AI 回复 | 模型的输出（通常由模型生成） |

</div>

&emsp;&emsp;下面是一个使用不同消息类型的示例：

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 构建一个对话
messages = [
    # 系统消息：定义 AI 的角色
    SystemMessage(content="你是一位专业的文档审核专家，擅长发现文档中的问题。"),

    # 用户消息：提供要审核的内容
    HumanMessage(content="请检查这句话是否有问题：'我们的产品必须满足所有客户需求。'"),
]

# 调用模型
response = llm.invoke(messages)
print(f"审核结果：\n{response.content}")
```

&emsp;&emsp;可以看到，通过设置 `SystemMessage`，我们成功让模型扮演了「文档审核专家」的角色。这就是<font color=red>提示词工程（Prompt Engineering）</font>的核心思想。

## 4. 文档审核中的结构化输出

&emsp;&emsp;在上面的示例中，大模型返回的是自然语言文本。虽然人类可以理解，但对于程序来说，这种非结构化的输出很难处理。

&emsp;&emsp;想象一下，如果我们要构建一个完整的文档审核系统，我们需要：

1. 将每个问题单独显示在界面上；
2. 按问题类型进行分类统计；
3. 将结果存储到数据库中；

&emsp;&emsp;这些都需要<font color=red>结构化的数据</font>，而不是一大段文本。

<div align=center><img src="Part 2. 基于 LangChain1.1 从零搭建AI 文档审核系统.assets/remote/202512171512397-d165979960.png" width=80%></div>

&emsp;&emsp;所以，我们需要用到 LangChain 提供的 `PydanticOutputParser`，可以让模型输出结构化的 JSON 数据，并自动解析为 Python 对象。

### 4.1 定义输出数据结构

&emsp;&emsp;首先，我们需要使用 Pydantic 定义输出的数据结构。Pydantic 是一个强大的数据验证库，它可以帮助我们定义数据模型并自动进行类型检查。对于文档审核，我们需要定义两个模型：

1. **ReviewIssue**：单个问题的数据结构；
2. **ReviewOutput**：包含问题列表的输出结构；

```python
from pydantic import BaseModel, Field
from typing import List, Literal

# 定义问题类型（使用 Literal 限制可选值）
IssueType = Literal["语法错误", "用词不当", "逻辑问题", "敏感表述"]

class ReviewIssue(BaseModel):
    """
    单个审核问题的数据结构。

    Field 装饰器用于添加字段描述，这些描述会被包含在给模型的提示中，
    帮助模型理解每个字段的含义。
    """
    type: str = Field(description="问题类型，如：语法错误、用词不当、逻辑问题、敏感表述")
    text: str = Field(description="问题所在的原文片段")
    explanation: str = Field(description="问题的详细说明")
    suggested_fix: str = Field(description="修改建议")
    para_index: int = Field(description="问题所在段落的索引（从0开始）")


class ReviewOutput(BaseModel):
    """
    审核输出的数据结构，包含问题列表。
    """
    issues: List[ReviewIssue] = Field(description="发现的问题列表")


# 展示数据结构
print("ReviewIssue 字段：")
for name, field in ReviewIssue.model_fields.items():
    print(f"  - {name}: {field.description}")
```

&emsp;&emsp;`Field(description=...)` 的作用非常重要：它不仅是代码文档，更会被<font color=red>包含在发送给模型的提示中</font>，帮助模型理解我们期望的输出格式。

### 4.2 创建输出解析器

&emsp;&emsp;接下来，我们创建一个 `PydanticOutputParser`，它会：

1. 自动生成格式说明（告诉模型应该输出什么格式）；
2. 将模型的输出解析为 Python 对象；

```python
from langchain_core.output_parsers import PydanticOutputParser

# 创建解析器
parser = PydanticOutputParser(pydantic_object=ReviewOutput)

# 查看生成的格式说明
format_instructions = parser.get_format_instructions()
print("格式说明（会发送给模型）：")
print("-" * 50)
print(format_instructions[:1000] + "...")
```

&emsp;&emsp;可以看到，解析器自动生成了详细的 JSON Schema，告诉模型应该输出什么格式的数据。这就是 LangChain 结构化输出的魔法！

## 5. 构建文档审核 Pipeline

### 5.1 设计系统提示词

&emsp;&emsp;一个好的系统提示词（System Prompt）是构建有效 Agent 应用的关键。在文档审核场景中，我们需要告诉大模型：

1. **它的角色**：专业的文档审核专家；
2. **任务目标**：识别文档中的问题；
3. **问题类型**：应该关注哪些类型的问题；
4. **输出格式**：如何输出结果；

&emsp;&emsp;下面是我们精心设计的系统提示词：

```python
SYSTEM_PROMPT = """你是一位专业的文档审核专家。
请仔细审查提供的文本，识别其中的问题。

需要检查的问题类型：
- 语法错误：错别字、标点符号错误、语病等
- 用词不当：使用了不恰当的词语或表达
- 敏感表述：使用了"必须"、"保证"、"一定"、"绝对"等过度承诺的措辞

注意事项：
1. 文档可能是中文或英文，请根据语言选择合适的审核标准
2. 使用输入中提供的段落索引（如 [0], [1], ...）来标识问题位置
3. 每个问题都需要提供具体的修改建议
4. 如果没有发现问题，返回空的问题列表
5. 按照要求的 JSON 格式输出结果
"""

print("系统提示词：")
print(SYSTEM_PROMPT)
```

### 5.2 构建用户提示词模板

&emsp;&emsp;用户提示词需要包含两部分：

1. **待审核的文本内容**：每个段落带上索引标记；
2. **输出格式说明**：告诉模型如何格式化输出；

```python
def build_user_prompt(paragraphs: list[dict], parser: PydanticOutputParser) -> str:
    """
    构建用户提示词。

    参数:
        paragraphs: 段落列表
        parser: 输出解析器

    返回:
        格式化的用户提示词
    """
    # 将段落格式化为带索引的文本
    formatted_text = "\n".join([
        f"[{i}] {p['content']}"
        for i, p in enumerate(paragraphs)
    ])

    # 构建完整的用户提示词
    user_prompt = f"""请审核以下文本内容：

{formatted_text}

如果发现问题，请按以下格式输出；如果没有问题，返回空的 issues 列表。

{parser.get_format_instructions()}
"""
    return user_prompt

# 测试
user_prompt = build_user_prompt(sample_paragraphs, parser)
print("用户提示词示例：")
print("-" * 50)
print(user_prompt[:800] + "...")
```

### 5.3 实现核心审核函数

&emsp;&emsp;现在，让我们把所有组件整合在一起，实现核心的文档审核函数。这是整个系统最关键的部分！

```python
from langchain_core.messages import SystemMessage, HumanMessage

def review_document(paragraphs: list[dict], llm: ChatOpenAI) -> ReviewOutput:
    """
    使用 LLM 审核文档内容。

    参数:
        paragraphs: 段落列表
        llm: LLM 模型实例

    返回:
        ReviewOutput 对象，包含发现的问题列表
    """
    # 创建输出解析器
    parser = PydanticOutputParser(pydantic_object=ReviewOutput)

    # 构建消息
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_user_prompt(paragraphs, parser)),
    ]

    # 调用 LLM
    print("正在调用 LLM 进行审核...")
    response = llm.invoke(messages)

    # 解析输出
    try:
        result = parser.parse(response.content)
        print(f"审核完成，发现 {len(result.issues)} 个问题")
        return result
    except Exception as e:
        print(f"解析输出失败: {e}")
        print(f"原始输出: {response.content[:500]}...")
        # 返回空结果
        return ReviewOutput(issues=[])
```

&emsp;&emsp;这个函数的核心流程非常清晰：

1. **创建解析器**：用于将模型输出解析为结构化对象
2. **构建消息**：包括系统提示词和用户提示词
3. **调用模型**：发送消息并获取响应
4. **解析结果**：将文本响应转换为 Python 对象

&emsp;&emsp;让我们来测试一下！

```python
# 使用示例数据测试
result = review_document(sample_paragraphs, llm)

# 展示结果
print("\n" + "=" * 60)
print("审核结果：")
print("=" * 60)

if result.issues:
    for i, issue in enumerate(result.issues, 1):
        print(f"\n问题 {i}：")
        print(f"   类型：{issue.type}")
        print(f"   位置：段落 [{issue.para_index}]")
        print(f"   原文：{issue.text}")
        print(f"   说明：{issue.explanation}")
        print(f"   建议：{issue.suggested_fix}")
else:
    print("未发现问题")
```

<div align=center><img src="Part 2. 基于 LangChain1.1 从零搭建AI 文档审核系统.assets/remote/202512171512397-d165979960.png" width=80%></div>

&emsp;&emsp;接下来，我们学习如何支持自定义审核规则，让系统更加灵活。

## 6. 自定义审核规则</center>

&emsp;&emsp;不同行业、不同场景对文档的审核要求是不同的，比如：

- **金融行业**：需要检查是否有违规承诺收益的表述；
- **医疗行业**：需要检查是否有夸大疗效的宣传；
- **法律文件**：需要检查条款是否有歧义；
- **营销文案**：需要检查是否有违反广告法的表述；

&emsp;&emsp;固定的审核规则无法满足所有需求，因此我们需要<font color=red>支持用户自定义审核规则</font>。

&emsp;&emsp;实现自定义规则的核心思想是：<font color=red>动态构建 Prompt</font>。我们将用户定义的规则动态注入到系统提示词中，让模型按照新的规则进行审核。

&emsp;&emsp;首先，我们需要定义规则的数据结构：

```python
from typing import Optional
from enum import Enum

class RiskLevel(str, Enum):
    """风险等级"""
    HIGH = "高"
    MEDIUM = "中"
    LOW = "低"


class RuleExample(BaseModel):
    """规则示例"""
    text: str = Field(description="示例文本")
    explanation: str = Field(description="说明")


class ReviewRule(BaseModel):
    """自定义审核规则"""
    name: str = Field(description="规则名称")
    description: str = Field(description="规则描述，说明什么情况下触发此规则")
    risk_level: RiskLevel = Field(description="风险等级")
    examples: list[RuleExample] = Field(default=[], description="示例列表")


# 创建一些示例规则
sample_rules = [
    ReviewRule(
        name="夸大宣传",
        description="检查是否有夸大产品效果或功能的表述，如'最好'、'第一'、'独家'等",
        risk_level=RiskLevel.HIGH,
        examples=[
            RuleExample(text="我们的产品是市场上最好的", explanation="使用了绝对化用语'最好'"),
            RuleExample(text="独家技术，行业第一", explanation="使用了'独家'、'第一'等夸大词汇"),
        ]
    ),
    ReviewRule(
        name="数据引用",
        description="检查引用的数据是否标注了来源",
        risk_level=RiskLevel.MEDIUM,
        examples=[
            RuleExample(text="据统计，90%的用户表示满意", explanation="未标注统计数据的来源"),
        ]
    ),
]

print("示例规则：")
for rule in sample_rules:
    print(f"\n {rule.name} [{rule.risk_level.value}风险]")
    print(f"     描述：{rule.description}")
```

&emsp;&emsp;接下来，我们需要修改提示词构建函数，支持动态注入自定义规则：

```python
def build_system_prompt(custom_rules: list[ReviewRule] = None) -> str:
    """
    构建系统提示词，支持自定义规则。

    参数:
        custom_rules: 自定义规则列表

    返回:
        系统提示词
    """
    # 基础规则类型
    issue_types = [
        "- 语法错误：错别字、标点符号错误、语病等",
        "- 用词不当：使用了不恰当的词语或表达",
        "- 敏感表述：使用了'必须'、'保证'、'一定'等过度承诺的措辞",
    ]

    # 动态添加自定义规则
    if custom_rules:
        for rule in custom_rules:
            rule_desc = f"- {rule.name}：{rule.description}"
            # 添加示例（如果有的话）
            if rule.examples:
                examples_str = "；".join([f'"{ex.text}"' for ex in rule.examples[:2]])
                rule_desc += f"（示例：{examples_str}）"
            issue_types.append(rule_desc)

    # 构建完整的系统提示词
    prompt = f"""你是一位专业的文档审核专家。
请仔细审查提供的文本，识别其中的问题。

需要检查的问题类型：
{chr(10).join(issue_types)}

注意事项：
1. 文档可能是中文或英文，请根据语言选择合适的审核标准
2. 使用输入中提供的段落索引（如 [0], [1], ...）来标识问题位置
3. 每个问题都需要提供具体的修改建议
4. 如果没有发现问题，返回空的问题列表
5. 按照要求的 JSON 格式输出结果
"""
    return prompt


# 测试：查看添加自定义规则后的提示词
custom_prompt = build_system_prompt(sample_rules)
print("含自定义规则的系统提示词：")
print("-" * 50)
print(custom_prompt)
```

&emsp;&emsp;可以看到，自定义规则已经被成功注入到系统提示词中。现在让我们更新审核函数，支持自定义规则：

```python
def review_document_with_rules(
    paragraphs: list[dict],
    llm: ChatOpenAI,
    custom_rules: list[ReviewRule] = None
) -> ReviewOutput:
    """
    使用 LLM 审核文档内容，支持自定义规则。

    参数:
        paragraphs: 段落列表
        llm: LLM 模型实例
        custom_rules: 自定义规则列表（可选）

    返回:
        ReviewOutput 对象
    """
    # 创建输出解析器
    parser = PydanticOutputParser(pydantic_object=ReviewOutput)

    # 构建消息（使用动态生成的系统提示词）
    messages = [
        SystemMessage(content=build_system_prompt(custom_rules)),
        HumanMessage(content=build_user_prompt(paragraphs, parser)),
    ]

    # 调用 LLM
    print("正在调用 LLM 进行审核...")
    if custom_rules:
        print(f"   使用自定义规则：{[r.name for r in custom_rules]}")

    response = llm.invoke(messages)

    # 解析输出
    try:
        result = parser.parse(response.content)
        print(f"审核完成，发现 {len(result.issues)} 个问题")
        return result
    except Exception as e:
        print(f"解析输出失败: {e}")
        return ReviewOutput(issues=[])
```

&emsp;&emsp;让我们用一段包含「夸大宣传」和「数据引用」问题的文本来测试：

```python
# 准备测试数据
test_paragraphs = [
    {"content": "我们的产品是市场上最先进、最可靠的解决方案。", "page_num": 1},
    {"content": "根据调查显示，95%的用户对我们的服务表示满意。", "page_num": 1},
    {"content": "公司成立于2020年，致力于人工智能技术的研发与应用。", "page_num": 2},
]

# 使用自定义规则进行审核
result = review_document_with_rules(test_paragraphs, llm, custom_rules=sample_rules)

# 展示结果
print("\n" + "=" * 60)
print("审核结果（使用自定义规则）：")
print("=" * 60)

if result.issues:
    for i, issue in enumerate(result.issues, 1):
        print(f"\n问题 {i}：")
        print(f"   类型：{issue.type}")
        print(f"   位置：段落 [{issue.para_index}]")
        print(f"   原文：{issue.text}")
        print(f"   说明：{issue.explanation}")
        print(f"   建议：{issue.suggested_fix}")
else:
    print("未发现问题")
```

## 7. 分块处理大文档

&emsp;&emsp;大语言模型有<font color=red>上下文长度限制</font>（Context Window）。如果文档内容超过了模型的上下文限制，我们需要将文档分成多个小块，分别进行审核，最后合并结果。

&emsp;&emsp;分块策略有很多种，最简单的是<font color=red>按段落数量分块</font>。

```python
def chunk_paragraphs(paragraphs: list[dict], chunk_size: int = 20) -> list[list[dict]]:
    """
    将段落列表分成多个小块。

    参数:
        paragraphs: 段落列表
        chunk_size: 每块的段落数量

    返回:
        分块后的段落列表
    """
    chunks = []
    for i in range(0, len(paragraphs), chunk_size):
        chunk = paragraphs[i:i + chunk_size]
        chunks.append(chunk)
    return chunks


# 演示分块效果
sample_data = [{"content": f"段落 {i}", "page_num": 1} for i in range(50)]
chunks = chunk_paragraphs(sample_data, chunk_size=15)

print(f"共 {len(sample_data)} 个段落，分成 {len(chunks)} 块")
for i, chunk in enumerate(chunks):
    print(f"   第 {i+1} 块：{len(chunk)} 个段落")
```

&emsp;&emsp;对于大文档，我们可以采用<font color=red>流式处理</font>的方式：逐块审核，实时返回结果。这样用户可以尽早看到部分审核结果，而不需要等待整个文档审核完成。

```python
from typing import Generator

def stream_review_document(
    paragraphs: list[dict],
    llm: ChatOpenAI,
    custom_rules: list[ReviewRule] = None,
    chunk_size: int = 20
) -> Generator[ReviewOutput, None, None]:
    """
    流式审核文档，逐块处理并yield结果。

    参数:
        paragraphs: 段落列表
        llm: LLM 模型实例
        custom_rules: 自定义规则列表
        chunk_size: 每块的段落数量

    Yields:
        每块的审核结果
    """
    # 分块
    chunks = chunk_paragraphs(paragraphs, chunk_size)
    total_chunks = len(chunks)

    print(f"文档共 {len(paragraphs)} 个段落，分成 {total_chunks} 块处理")
    print("-" * 50)

    # 创建解析器
    parser = PydanticOutputParser(pydantic_object=ReviewOutput)
    system_prompt = build_system_prompt(custom_rules)

    # 逐块处理
    for i, chunk in enumerate(chunks):
        print(f"\n正在处理第 {i+1}/{total_chunks} 块...")

        # 构建消息
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=build_user_prompt(chunk, parser)),
        ]

        # 调用 LLM
        response = llm.invoke(messages)

        # 解析结果
        try:
            result = parser.parse(response.content)
            print(f"   发现 {len(result.issues)} 个问题")
            yield result
        except Exception as e:
            print(f"   解析失败: {e}")
            yield ReviewOutput(issues=[])

    print("\n" + "-" * 50)
    print("全部处理完成！")
```

&emsp;&emsp;让我们用一个较长的测试文档来演示流式审核的效果：

```python
# 准备一个较长的测试文档
long_document = [
    {"content": "本公司是行业内最具实力的企业之一。", "page_num": 1},
    {"content": "我们承诺必须在30天内完成项目交付。", "page_num": 1},
    {"content": "根据内部调查，客户满意度达到98%。", "page_num": 1},
    {"content": "公司拥有一支优秀的技术团队。", "page_num": 2},
    {"content": "我们的解决方案能够完全满足所有需求。", "page_num": 2},
    {"content": "产品采用了世界领先的技术架构。", "page_num": 2},
]

# 流式审核（设置较小的chunk_size以演示效果）
all_issues = []
for result in stream_review_document(long_document, llm, chunk_size=3):
    all_issues.extend(result.issues)

# 汇总结果
print("\n" + "=" * 60)
print(f"汇总：共发现 {len(all_issues)} 个问题")
print("=" * 60)

for i, issue in enumerate(all_issues, 1):
    print(f"\n{i}. [{issue.type}] {issue.text[:30]}...")
    print(f"   建议：{issue.suggested_fix}")
```

# 8. 完整运行示例

&emsp;&emsp;最后，我们将所有组件整合在一起，实现一个完整的文档审核流程：

```python
def review_pdf_document(
    pdf_path: str,
    deepseek_api_key: str,
    custom_rules: list[ReviewRule] = None,
    chunk_size: int = 20,
) -> list[ReviewIssue]:
    """
    完整的 PDF 文档审核流程。

    参数:
        pdf_path: PDF 文件路径
        deepseek_api_key: DeepSeek API 密钥
        custom_rules: 自定义规则列表
        chunk_size: 分块大小

    返回:
        发现的所有问题列表
    """
    print("开始文档审核流程")
    print("=" * 60)

    # 步骤1：解析 PDF
    print("\n步骤1：解析 PDF 文档")
    paragraphs = extract_text_from_pdf(pdf_path)
    print(f"提取了 {len(paragraphs)} 页内容")

    # 将页面内容转换为段落格式
    all_paragraphs = []
    for page in paragraphs:
        # 按换行符分割段落
        for para in page["content"].split("\n"):
            if para.strip():
                all_paragraphs.append({
                    "content": para.strip(),
                    "page_num": page["page_num"]
                })
    print(f"  共 {len(all_paragraphs)} 个段落")

    # 步骤2：初始化 LLM
    print("\n步骤2：初始化 LLM 模型")
    llm = init_llm(deepseek_api_key)
    print("   模型初始化完成")

    # 步骤3：执行审核
    print("\n步骤3：执行文档审核")
    all_issues = []
    for result in stream_review_document(all_paragraphs, llm, custom_rules, chunk_size):
        all_issues.extend(result.issues)

    # 汇总结果
    print("\n" + "=" * 60)
    print(f"审核完成！共发现 {len(all_issues)} 个问题")

    return all_issues
```

```python
# 使用示例（需要提供实际的 PDF 文件）
issues = review_pdf_document(
    pdf_path="./data/LangChain v1.1 文档审核类Agent开发实战.pdf",
    deepseek_api_key=DEEPSEEK_API_KEY,
    custom_rules=sample_rules,
    chunk_size=20
)

print(issues)
```

&emsp;&emsp;我们还可以将审核结果输出为格式化的报告：

```python
def generate_report(issues: list[ReviewIssue], output_path: str = None) -> str:
    """
    生成审核报告。

    参数:
        issues: 问题列表
        output_path: 输出文件路径（可选）

    返回:
        报告文本
    """
    # 按类型分组统计
    type_counts = {}
    for issue in issues:
        type_counts[issue.type] = type_counts.get(issue.type, 0) + 1

    # 生成报告
    report = []
    report.append("# 文档审核报告")
    report.append("")
    report.append(f"## 概要")
    report.append(f"- 发现问题总数：**{len(issues)}**")
    report.append("")
    report.append("## 问题类型分布")
    for issue_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        report.append(f"- {issue_type}：{count} 个")
    report.append("")
    report.append("## 问题详情")
    for i, issue in enumerate(issues, 1):
        report.append(f"\n### 问题 {i}")
        report.append(f"- **类型**：{issue.type}")
        report.append(f"- **位置**：段落 {issue.para_index}")
        report.append(f"- **原文**：{issue.text}")
        report.append(f"- **说明**：{issue.explanation}")
        report.append(f"- **建议**：{issue.suggested_fix}")

    report_text = "\n".join(report)

    # 保存到文件（如果指定了路径）
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"报告已保存到：{output_path}")

    return report_text
```

```python
# 演示：使用之前的审核结果生成报告
if all_issues:
    report = generate_report(all_issues)
    print(report)
```

```json
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                           Step 1: 准备真实 PDF 文档                           │
    │  ├─ 输入：tutorials/data/LangChain v1.1 文档审核类Agent开发实战.pdf             │
    │  └─ 说明：真实 PDF                                    │
    └─────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                      Step 2: MinerU 智能解析（核心步骤）                       │
    │                                                                                  │
    │  ┌──────────────────────────────────────────────────────────────────────────┐   │
    │  │ 2.1 请求上传 URL                                                          │   │
    │  │     POST /api/v4/file-urls/batch                                          │   │
    │  │     ├─ 请求体：{files: [{name, data_id}], model_version}                  │   │
    │  │     └─ 返回值：{batch_id, upload_url}                                     │   │
    │  └──────────────────────────────────────────────────────────────────────────┘   │
    │                                        │                                         │
    │                                        ▼                                         │
    │  ┌──────────────────────────────────────────────────────────────────────────┐   │
    │  │ 2.2 上传 PDF 文件                                                         │   │
    │  │     PUT {upload_url}                                                       │   │
    │  │     └─ 请求体：PDF 文件的二进制内容                                        │   │
    │  └──────────────────────────────────────────────────────────────────────────┘   │
    │                                        │                                         │
    │                                        ▼                                         │
    │  ┌──────────────────────────────────────────────────────────────────────────┐   │
    │  │ 2.3 轮询解析状态                                                          │   │
    │  │     GET /api/v4/extract-results/batch/{batch_id}                          │   │
    │  │     ├─ 状态：pending → processing → done/failed                           │   │
    │  │     └─ 成功后返回：{full_zip_url}                                         │   │
    │  └──────────────────────────────────────────────────────────────────────────┘   │
    │                                        │                                         │
    │                                        ▼                                         │
    │  ┌──────────────────────────────────────────────────────────────────────────┐   │
    │  │ 2.4 下载并解析 ZIP 产物                                                   │   │
    │  │     GET {full_zip_url}                                                     │   │
    │  │     ZIP 包含：                                                             │   │
    │  │     ├─ layout.json      → 布局信息（行/span/bbox，用于精确高亮）           │   │
    │  │     ├─ content.json     → 抽取的文本内容                                   │   │
    │  │     └─ page_*.png/jpg   → 页面渲染图（用于推断 canvas 尺寸）               │   │
    │  └──────────────────────────────────────────────────────────────────────────┘   │
    │                                        │                                         │
    │                                        ▼                                         │
    │  ┌──────────────────────────────────────────────────────────────────────────┐   │
    │  │ 2.5 归一化段落数据                                                        │   │
    │  │     输出格式：                                                             │   │
    │  │     [                                                                      │   │
    │  │       {                                                                    │   │
    │  │         "content": "段落文本内容",                                         │   │
    │  │         "page_num": 1,                                                     │   │
    │  │         "bbox": [x1, y1, x2, y2],  # 边界框坐标                            │   │
    │  │         "canvas_size": [width, height]  # 页面尺寸                         │   │
    │  │       },                                                                   │   │
    │  │       ...                                                                  │   │
    │  │     ]                                                                      │   │
    │  └──────────────────────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                    Step 3: LangChain 文档审核 Pipeline                        │
    │                                                                                  │
    │  ┌──────────────────────────────────────────────────────────────────────────┐   │
    │  │ 3.1 定义审核规则                                                          │   │
    │  │     预设规则：                                                             │   │
    │  │     ├─ Grammar & Spelling（语法与拼写）                                    │   │
    │  │     └─ Definitive Language（绝对化表述）                                   │   │
    │  │     自定义规则：                                                           │   │
    │  │     └─ {name, description, risk_level, examples}                          │   │
    │  └──────────────────────────────────────────────────────────────────────────┘   │
    │                                        │                                         │
    │                                        ▼                                         │
    │  ┌──────────────────────────────────────────────────────────────────────────┐   │
    │  │ 3.2 构建排除规则（避免误判）                                              │   │
    │  │     不应报告为错误的情况：                                                 │   │
    │  │     ├─ 序号：1、2、(1)、①、一、二 等                                       │   │
    │  │     ├─ 日期占位符：年/月/日、____年____月____日                            │   │
    │  │     ├─ 金额占位符：___元、____元整                                         │   │
    │  │     ├─ 勾选框：口、□、☐、○ 等                                             │   │
    │  │     └─ 合同模板字段：甲方、乙方、签字盖章 等                               │   │
    │  └──────────────────────────────────────────────────────────────────────────┘   │
    │                                        │                                         │
    │                                        ▼                                         │
    │  ┌──────────────────────────────────────────────────────────────────────────┐   │
    │  │ 3.3 分块处理（Chunking）                                                  │   │
    │  │     ├─ 为什么需要分块？避免一次性给模型太长的文本                          │   │
    │  │     ├─ 分块策略：每 N 个段落为一个 chunk                                   │   │
    │  │     └─ 每个 chunk 独立审核，最后合并结果                                   │   │
    │  └──────────────────────────────────────────────────────────────────────────┘   │
    │                                        │                                         │
    │                                        ▼                                         │
    │  ┌──────────────────────────────────────────────────────────────────────────┐   │
    │  │ 3.4 LLM 审核调用                                                          │   │
    │  │     输入：                                                                 │   │
    │  │     ├─ System Prompt（角色定义 + 规则说明 + 排除规则）                     │   │
    │  │     └─ Human Message（段落内容 [0],[1],... + 输出格式要求）                │   │
    │  │     输出（结构化 JSON）：                                                  │   │
    │  │     {                                                                      │   │
    │  │       "issues": [                                                          │   │
    │  │         {                                                                  │   │
    │  │           "type": "问题类型",                                              │   │
    │  │           "text": "问题文本片段",                                          │   │
    │  │           "para_index": 0,                                                 │   │
    │  │           "explanation": "问题解释",                                       │   │
    │  │           "suggested_fix": "修改建议"                                      │   │
    │  │         }                                                                  │   │
    │  │       ]                                                                    │   │
    │  │     }                                                                      │   │
    │  └──────────────────────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                          Step 4: 输出问题列表                                 │
    │  最终产出：                                                                      │
    │  ┌────────────┬──────────────┬────────────┬──────────────┬────────────────┐     │
    │  │ page_num   │ type         │ text       │ explanation  │ suggested_fix  │     │
    │  ├────────────┼──────────────┼────────────┼──────────────┼────────────────┤     │
    │  │ 1          │ Grammar      │ 错误片段   │ 问题解释     │ 修改建议       │     │
    │  │ 2          │ 自定义规则   │ ...        │ ...          │ ...            │     │
    │  └────────────┴──────────────┴────────────┴──────────────┴────────────────┘     │
    │  可用于：前端高亮展示、生成审核报告、人工复核                                    │
    └─────────────────────────────────────────────────────────────────────────────────┘
```
