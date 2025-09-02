# LLM・データ処理機能 設計書

## 1. 概要

本システムでは、ローカル環境でのテスト（Ollama利用）と、本番環境（Azure OpenAI利用）のスムーズな切り替えを可能にするため、柔軟性の高いLLM（大規模言語モデル）管理機能を実装する。

この設計は [MochiRAG/core/llm_manager.py](https://github.com/chottokun/MochiRAG/blob/main/core/llm_manager.py) のアーキテクチャを参考に、設定ファイル駆動でLLMの実装を抽象化・カプセル化することを目的とする。

## 2. アーキテクチャ

主要コンポーネントは以下の2つで構成される。

-   **`ConfigManager`**: 設定ファイル (`config.toml`) からLLMの接続情報などを読み込む責務を持つ。
-   **`LLMManager`**: `ConfigManager`から得た設定に基づき、要求されたLLMのクライアントインスタンスを生成・管理する責務を持つ。

### 2.1. `LLMManager` (シングルトン)

-   **設計パターン**: シングルトンパターンを採用する。
-   **目的**:
    -   アプリケーション全体で `LLMManager` のインスタンスを唯一に保つことで、LLMクライアントの一元的な管理を実現する。
    -   LLMクライアントのインスタンスをキャッシュし、不要な再生成を防ぐことで、リソースの効率的な利用とパフォーマンスの向上を図る。
-   **主要メソッド**:
    -   `get_llm(name: str | None) -> BaseChatModel`:
        -   引数 `name` で指定されたLLMのインスタンスを返す。
        -   `name` が `None` の場合は、設定ファイルで指定されたデフォルトのLLMを返す。
        -   インスタンスが未生成の場合は、`ConfigManager` から設定を読み込み、適切なLLMクライアント（例: `ChatOllama`, `AzureChatOpenAI`）を初期化して返す。
        -   生成済みのインスタンスは内部でキャッシュし、2回目以降の呼び出しではキャッシュしたインスタンスを返す。

### 2.2. `ConfigManager`

-   **目的**: LLMに関する設定をコードから完全に分離する。
-   **機能**:
    -   プロジェクトルートの `config.toml` ファイルを読み込む。
    -   デフォルトで使用するLLMの名前や、各LLMプロバイダー（Ollama, Azure）ごとの設定（モデル名、APIキー、ベースURLなど）へのアクセスを提供する。

## 3. 設定ファイル (`config.toml`) の構造

TOML形式を採用し、可読性と拡張性の高い設定管理を行う。

```toml
# デフォルトで使用するLLMの名前
default_llm = "ollama_llama3"

# LLMプロバイダーごとの設定
[llm.ollama_llama3]
provider = "ollama"
model_name = "llama3"
base_url = "http://localhost:11434"

[llm.azure_gpt4]
provider = "azure"
model_name = "gpt-4o"
# AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT などの環境変数から読み込む想定
```

## 4. 利用例

`LLMManager` を通じて、以下のように簡単にLLMインスタンスを取得できる。

```python
from rss_mcp.llm_manager import llm_manager

# デフォルトのLLM (ollama_llama3) を取得
ollama_llm = llm_manager.get_llm()

# 名前を指定してAzureのLLMを取得
azure_llm = llm_manager.get_llm("azure_gpt4")
```

---

## 5. データ処理フロー

`requirement.md` に基づき、非構造化ドキュメント（まずはMarkdown）をナレッジグラフに変換するための前処理を行う。この責務を担うのが `DocumentProcessor` である。

### 5.1. `DocumentProcessor`

- **目的**: 指定されたドキュメントファイルをロードし、LLMが処理しやすいように意味的なまとまり（チャンク）に分割する。
- **主要メソッド**:
    - `process_file(file_path: str) -> List[Document]`:
        -   引数 `file_path` で指定されたMarkdownファイルをロードする。
        -   内部で `langchain_community.document_loaders.UnstructuredMarkdownLoader` を使用する。
        -   ロードしたドキュメントを、`langchain.text_splitter.RecursiveCharacterTextSplitter` を用いてチャンクに分割する。
        -   分割された `Document` オブジェクトのリストを返す。
- **設定**: チャンクサイズやオーバーラップなどのパラメータは、将来的に `config.toml` から読み込めるように拡張可能にする。

### 5.2. 利用例

```python
from rss_mcp.document_processor import DocumentProcessor

# プロセッサを初期化
processor = DocumentProcessor()

# ファイルを処理してチャンクを取得
chunks = processor.process_file("path/to/your/document.md")
```

### 5.3. `GraphConverter`

- **目的**: 分割されたドキュメントチャンクを、`LLMGraphTransformer` を用いてナレッジグラフ形式 (`GraphDocument`) に変換する。
- **依存関係**: `LLMManager` から適切なLLMインスタンスを取得する。
- **主要メソッド**:
    - `convert_to_graph(documents: List[Document]) -> List[GraphDocument]`:
        -   引数 `documents` でチャンクのリストを受け取る。
        -   `LLMManager` を通じて、グラフ構築用のLLM（例: `ollama_llama3`）を取得する。
        -   `LLMGraphTransformer` を取得したLLMで初期化する。
        -   `transformer.convert_to_graph_documents()` を呼び出し、チャンクを `GraphDocument` オブジェクトのリストに変換して返す。

### 5.4. 利用例

```python
from rss_mcp.graph_converter import GraphConverter

# コンバーターを初期化
# (内部でLLMManagerが利用される)
converter = GraphConverter()

# チャンクをグラフドキュメントに変換
graph_documents = converter.convert_to_graph(chunks)
```
