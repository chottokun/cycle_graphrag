# LLM管理機能 設計書

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
