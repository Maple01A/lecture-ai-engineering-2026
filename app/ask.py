"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


@dataclass(frozen=True)
class BenchmarkResult:
    """1回のAPI呼び出しで得られた測定値。"""

    model: str
    question: str
    elapsed_seconds: float | None
    input_tokens: int | None
    output_tokens: int | None
    error: str | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


def _usage_counts(response: Any) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None, None
    return (
        getattr(usage, "prompt_token_count", None),
        getattr(usage, "candidates_token_count", None),
    )


def ask(question: str, model: str = MODEL, client_factory: Callable[..., Any] = genai.Client) -> str:
    client = client_factory(api_key=get_api_key())

    # 会話を作成
    chat = client.chats.create(model=model)

    # 最初の質問
    response = chat.send_message(question)

    usage = response.usage_metadata
    print(
        f"[usage] input={usage.prompt_token_count} "
        f"output={usage.candidates_token_count}"
    )

    print(response.text)

    # 続けて質問
    while True:
        question = input("\n質問（終了: exit）: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        response = chat.send_message(question)
        print(response.text)

    return response.text


def run_benchmark(
    models: Iterable[str],
    questions: Iterable[str],
    repeats: int = 1,
    client_factory: Callable[..., Any] = genai.Client,
) -> list[BenchmarkResult]:
    """モデル・質問の組み合わせを実行し、時間とトークン数を返す。"""
    if repeats < 1:
        raise ValueError("repeats は1以上で指定してください")

    model_list = list(models)
    question_list = list(questions)
    results: list[BenchmarkResult] = []
    for model in model_list:
        client = client_factory(api_key=get_api_key())
        for question in question_list:
            for _ in range(repeats):
                chat = client.chats.create(model=model)
                started = time.perf_counter()
                try:
                    response = chat.send_message(question)
                except errors.APIError as exc:
                    results.append(
                        BenchmarkResult(
                            model,
                            question,
                            time.perf_counter() - started,
                            None,
                            None,
                            api_error_message(exc.code, exc.message),
                        )
                    )
                    continue

                elapsed = time.perf_counter() - started
                input_tokens, output_tokens = _usage_counts(response)
                results.append(
                    BenchmarkResult(
                        model,
                        question,
                        elapsed,
                        input_tokens,
                        output_tokens,
                    )
                )
    return results


def _print_benchmark(results: Iterable[BenchmarkResult]) -> None:
    print("model\tquestion\tseconds\tinput_tokens\toutput_tokens\ttotal_tokens\tstatus")
    for result in results:
        elapsed = "-" if result.elapsed_seconds is None else f"{result.elapsed_seconds:.3f}"
        values = (
            result.model,
            result.question,
            elapsed,
            result.input_tokens if result.input_tokens is not None else "-",
            result.output_tokens if result.output_tokens is not None else "-",
            result.total_tokens if result.total_tokens is not None else "-",
            result.error or "ok",
        )
        print("\t".join(str(value).replace("\t", " ") for value in values))


def main() -> None:
    parser = argparse.ArgumentParser(description="会話履歴を保持してLLMに質問する")
    parser.add_argument("question", nargs="?", help="対話モードで送る質問")
    parser.add_argument("--benchmark", action="store_true", help="モデル・質問ごとの測定を行う")
    parser.add_argument("--model", dest="models", action="append", help="測定対象モデル（複数指定可）")
    parser.add_argument("--question", dest="questions", action="append", help="測定対象の質問（複数指定可）")
    parser.add_argument("--repeat", type=int, default=1, help="各組み合わせの実行回数（既定: 1）")
    args = parser.parse_args()

    try:
        if args.benchmark:
            models = args.models or [MODEL]
            questions = args.questions or ([args.question] if args.question else [DEFAULT_QUESTION])
            _print_benchmark(run_benchmark(models, questions, args.repeat))
        else:
            question = args.question or DEFAULT_QUESTION
            model = (args.models or [MODEL])[0]
            print(f"[model] {model}")
            print(f"[質問] {question}\n")
            ask(question, model=model)
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
