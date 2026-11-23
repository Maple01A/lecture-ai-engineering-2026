"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、登録されたツールを全部呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

import json

from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app.agent import agent
from app.retrieval import search_tree_graph

BASE_TOOLS = {"search_materials"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


def test_base_tools_are_registered_and_called(monkeypatch):
    monkeypatch.setattr("app.retrieval.search", lambda *args, **kwargs: [])
    model = TestModel()
    with agent.override(model=model):
        result = agent.run_sync("LoRAとは？")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert BASE_TOOLS <= offered  # 演習でツールを足しても壊れないよう「含む」で判定

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert BASE_TOOLS <= called
    # 承認が必要なツール（requires_approval=True）は、実行されずに承認待ちになるので戻り値が無い
    waiting = len(result.output.approvals) if isinstance(result.output, DeferredToolRequests) else 0
    assert len(_parts(result, ToolReturnPart)) + waiting == len(_parts(result, ToolCallPart))
    assert result.output


def test_material_result_reaches_the_model(monkeypatch):
    hit = {"page_id": "lec02-main-p052", "lecture": 2, "page": 52, "content": "元の重みは固定"}
    monkeypatch.setattr("app.retrieval.search", lambda *args, **kwargs: [hit])
    model = TestModel(call_tools=["search_materials"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_materials"
    assert json.loads(returns[0].content)["results"] == [hit]


def test_tree_graph_search_merges_graph_and_tree_siblings(monkeypatch):
    pages = [
        {"page_id": "lec03-main-p038", "document_id": "lec03-main", "lecture": 3,
         "document_title": "AIエージェント", "section": "Part 6 メモリ",
         "title": "メモリがないと毎回ゼロから", "page": 38, "content": "前回の失敗"},
        {"page_id": "lec03-main-p039", "document_id": "lec03-main", "lecture": 3,
         "document_title": "AIエージェント", "section": "Part 6 メモリ",
         "title": "実装例：Claude", "page": 39, "content": "コンテキスト内と外部保存先"},
        {"page_id": "lec03-main-p040", "document_id": "lec03-main", "lecture": 3,
         "document_title": "AIエージェント", "section": "Part 6 メモリ",
         "title": "実装例：OpenClaw", "page": 40, "content": "長期メモリ"},
    ]
    monkeypatch.setattr("app.retrieval.search", lambda *args, **kwargs: [pages[1]])
    monkeypatch.setattr(
        "app.retrieval.search_graph",
        lambda *args, **kwargs: {"seeds": [pages[1]], "expanded": [pages[2]],
                                 "max_hops": 2},
    )
    monkeypatch.setattr("app.retrieval.load_pages", lambda *args, **kwargs: pages)

    result = search_tree_graph("コンパクション メモリ")

    ids = [page["page_id"] for page in result["tree_graph"]]
    assert "lec03-main-p039" in ids
    assert "lec03-main-p040" in ids
    assert "lec03-main-p038" in ids
    assert len(ids) == len(set(ids))
