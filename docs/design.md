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

---

## 6. グラフデータベース格納

生成された `GraphDocument` オブジェクトを永続化するため、Neo4jデータベースに格納する。この責務を担うのが `GraphStore` である。

### 6.1. `GraphStore`

- **目的**: Neo4jデータベースへの接続を管理し、グラフドキュメントの書き込みを行う。
- **依存関係**: `ConfigManager` からNeo4jの接続情報（URI, ユーザ名, パスワード）を取得する。
- **主要メソッド**:
    - `__init__()`: `ConfigManager` から設定を読み込み、`langchain_community.graphs.Neo4jGraph` のインスタンスを初期化して保持する。
    - `save_graph(graph_documents: List[GraphDocument])`:
        -   引数で `GraphDocument` のリストを受け取る。
        -   保持している `Neo4jGraph` インスタンスの `add_graph_documents` メソッドを呼び出して、データをデータベースに書き込む。
    - `create_vector_index()`:
        -   `EmbeddingManager` を利用して、Embeddingモデルをロードする。
        -   Neo4jの各`Chunk`ノードに対して、その`text`プロパティからベクトル埋め込み（Embedding）を生成し、`embedding`プロパティとして設定する。
        -   `embedding`プロパティに対するベクトルインデックスをNeo4jデータベース上に作成する。これにより、高速な類似度検索が可能になる。

### 6.2. 利用例

```python
from rss_mcp.graph_store import GraphStore

# GraphStoreを初期化
# (内部でConfigManagerからDB設定を読み込む)
graph_store = GraphStore()

# グラフドキュメントをDBに保存
graph_store.save_graph(graph_documents)

# ベクトルインデックスを作成
graph_store.create_vector_index()
```

---

## 7. Embedding管理

### 7.1. `EmbeddingManager`

- **目的**: HuggingFaceの`sentence-transformers`モデルをロードし、アプリケーション全体で共有するための管理クラス。
- **設計パターン**: シングルトンパターンを採用。
- **主要メソッド**:
    - `get_model()`: `HuggingFaceEmbeddings`のインスタンスを返す。モデル名は`config.toml`から読み込む。

---

## 8. 対話型GraphRAG（ハイブリッド検索）

格納されたナレッジグラフに対し、ベクトル検索とグラフ検索を組み合わせたハイブリッドなアプローチで対話を行う。

### 8.1. `GraphRAGAgent`

- **目的**: ユーザーからの質問に対し、最適な検索戦略（ベクトル検索、グラフ検索）を組み合わせて実行し、精度の高い回答を生成する。
- **検索フロー**:
    1.  **ベクトル検索**: ユーザーの質問をベクトル化し、Neo4jのベクトルインデックスを使って関連性の高い`Chunk`ノードを検索する。
    2.  **グラフ検索**: 1で見つかった`Chunk`ノードの周辺にあるエンティティや、質問に含まれるキーワードを基に、グラフを走査（Cypherクエリ）して関連情報を収集する。
    3.  **情報統合と回答生成**: ベクトル検索とグラフ検索で得られた両方のコンテキストを統合し、LLMに渡して最終的な回答を生成する。
- **主要コンポーネント**:
    - `langchain_community.vectorstores.Neo4jVector` をリトリーバーとして利用。
    - `GraphCypherQAChain` またはカスタムのCypher生成チェーンを利用。
    - LangChain Expression Language (LCEL) を用いて、上記のコンポーネントを組み合わせたカスタムチェーンを構築する。
- **主要メソッド**:
    - `query(question: str) -> str`:
        -   引数でユーザーの質問を受け取る。
        -   内部で構築したハイブリッド検索チェーンを実行し、最終的な回答文字列を返す。

### 7.2. 利用例

```python
from rss_mcp.graph_rag_agent import GraphRAGAgent

# エージェントを初期化
# (内部でLLMManagerとGraphStoreが利用される)
agent = GraphRAGAgent()

# 質問して回答を得る
question = "Who is Jules?"
answer = agent.query(question)
print(answer)
```

---

## 9. グラフ可視化

Neo4jに格納されているナレッジグラフ全体を、インタラクティブに可視化する機能。

### 9.1. `GraphStore`の拡張

- **目的**: 可視化ライブラリ(`pyvis`)が扱いやすい形式で、グラフのノードとエッジの一覧を取得する。
- **主要メソッド**:
    - `get_graph_for_visualization() -> Tuple[List[Dict], List[Dict]]`:
        -   Neo4jからすべてのノードとリレーションシップを取得するCypherクエリを実行する。
        -   結果を、ノードのリストとエッジのリストに整形して返す。各ノード/エッジは`pyvis`が必要とする形式の辞書。

### 9.2. Streamlit可視化ページ

- **ファイル**: `pages/3_Visualize_Graph.py`
- **機能**:
    - `GraphStore`の`get_graph_for_visualization`を呼び出してグラフデータを取得する。
    - 取得したデータを用いて`pyvis.network.Network`オブジェクトを構築する。
    - `streamlit_agraph.agraph`関数に`pyvis`オブジェクトを渡し、Streamlitページ内にグラフを描画する。
    - （将来的な拡張）UI上に、表示するノードのラベルやリレーションのタイプをフィルタリングする機能を追加する。
