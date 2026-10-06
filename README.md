# Document Review Agent v2.0

> 基于 LangChain v1.1 的通用 AI 文档审核系统 v2.0，支持人机协同（HITL）、多类型文档、企业级部署。

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-v1.1-green)](https://www.langchain.com/)
[![React](https://img.shields.io/badge/React-18+-61dafb)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688)](https://fastapi.tiangolo.com/)
[![Terraform](https://img.shields.io/badge/Terraform-1.x+-purple)](https://www.terraform.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ 特性

- 🤖 **智能审核**：基于 LangChain v1.1 的 Agent 自动审核
- 👥 **人机协同（HITL）**：AI 初审 + 人工复核，确保准确率
- 📑 **多类型文档**：合同、票据、报告、标书等多场景
- 🔧 **可配置规则**：灵活的审核规则引擎
- 🏢 **企业级架构**：前后端分离、模块化设计
- ☁️ **云原生部署**：Terraform + Docker，支持一键上云
- 📊 **审核溯源**：每条审核结论可追溯到原文位置

---

## 🏗️ 技术架构

```
┌───────────────────────────────────────────────────────┐
│                       Frontend                        │
│                (React + TypeScript)                   │
│  ┌──────────┐ ┌────────────┐ ┌───────────────────┐   │
│  │ 文档上传 │ │ 审核工作台 │ │ 审核报告 & 溯源    │   │
│  └──────────┘ └────────────┘ └───────────────────┘   │
└──────────────────────────┬────────────────────────────┘
                           │
               ┌───────────▼────────────┐
               │      Backend API       │
               │      (FastAPI)         │
               └───────────┬────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼─────┐      ┌─────▼──────┐    ┌────▼─────┐
   │ LangChain │      │  规则引擎   │    │ 文档解析  │
   │  v1.1     │      │            │    │ (MinerU)  │
   │  Agent    │      │            │    │          │
   └──────────┘      └────────────┘    └──────────┘
```

---

## 📁 项目结构

```
.
├── app/                        # 主应用
│   ├── api/                    # API 层
│   └── ui/                     # 前端 UI
├── common/                     # 公共模块
│   ├── logger.py
│   └── models.py
├── docs/                       # 项目文档
│   ├── architecture.md
│   └── getting_started.md
├── eval/                       # 评测模块
│   └── config.json
├── flows/                      # 工作流定义
│   ├── requirements.txt
│   └── .env.tpl
├── infra/                      # 基础设施（Terraform）
│   ├── main.tf
│   ├── variables.tf
│   └── ...
├── tutorials/                  # 教程文档
│   ├── Part 1. 技术架构概览
│   ├── Part 2. 从零搭建系统
│   └── Part 3. 人机交互审核
├── Taskfile.yml                # 任务编排
├── install.sh / install.bat    # 安装脚本
├── start.sh / start.bat        # 启动脚本
├── stop.sh / stop.bat          # 停止脚本
└── README.md
```

> 💡 **示例数据**：MinerU 解析示例、教程数据等位于 `../data/` 目录。

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+
- 大模型 API Key
- Docker（推荐部署方式）

### 快速启动

```bash
# 一键安装
./install.sh

# 启动服务
./start.sh
```

### 手动启动

```bash
# 后端
cd app/api
pip install -r requirements.txt
uvicorn main:app --reload

# 前端
cd app/ui
npm install
npm run dev
```

访问 http://localhost:3000

### 云部署

使用 Terraform 一键部署到云端：

```bash
cd infra
terraform init
terraform apply
```

---

## 📚 相关课程

本项目配套三部分完整课程：

- Part 1：智能文档审核 Agent 技术架构概览
- Part 2：基于 LangChain1.1 从零搭建 AI 文档审核系统
- Part 3：基于 LangChain1.1 实现人机交互审核（HITL）

- 📖 课程文档：[飞书知识库](https://scnxinvxtnbo.feishu.cn/wiki/TqwqwU6AgiACIfksEjVcvEGSnob)
- 🎥 视频教程：[课程链接](#)（待补充）

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
