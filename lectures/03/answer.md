# 第3回 解答

## 演習

- 第1回（素のLLM）とエージェントの回答の違い：
  - 第1回のLLMは「3レベル」を一般的な課題の難易度を示す3段階として解釈しており、AIE2026における正しい回答とは異なっていた。 一方、エージェントは用語集を参照することで、AIE2026における課題の難易度3段階について正しく回答できた。
- `--trace` で見えたツール呼び出し（ツール名と引数）：
  - 「3レベル」の質問では、search_glossary({"query": "3レベル"}) が1回呼び出された。
  - 「LoRAとQLoRA」の質問では、search_glossary({"query": "LoRA"}) と search_glossary({"query": "QLoRA"}) がそれぞれ1回ずつ呼び出された。
- 1つの質問での LLM 呼び出し回数（`requests`）：
  - 「3レベル」の質問では、requests=2 で、ツール呼び出しは1回だった。
  - 「LoRAとQLoRA」の質問では、requests=2 で、ツール呼び出しは2回だった。
- 追加したツールと、docstring に書いた説明：
  - 追加したツールは make_quiz。docstring には、指定した回の用語集から確認問題とその回答を1問作成するツールと指定した。

## 宿題

- 取り組んだレベル：実装
- やったこと：
  - `app/tools.py` に `save_learning_log(entry)` を追加し、`learning_log/learning_log.jsonl` に日時と学習内容を追記するようにした。
  - `app/agent.py` では、`DeferredToolRequests` を `output_type` に追加し、保存ツールを `requires_approval=True` に設定して登録した。
  - CLIでは、保存前に `Yes/No` の確認を表示し、利用者の承認結果を `build_results()` に渡すようにした。
- 結果：
  - 保存ツールが呼び出されると、実行前に DeferredToolRequests による承認待ちの状態になった。
  - y を入力すると保存が実行され、learning_log/learning_log.jsonl に次のような記録が追加された。
{"timestamp": "...", "entry": "今日、LoRAについて学びました。"}
  - n を入力すると、保存を承認しないという結果を返すことができる。
- 気づき・疑問：
  - 保存のように取り消しが難しい処理については、ツールを実行する前に利用者の確認を挟むことで、安全性を高めることができる。また、自動保存機能がないため、手間はかかるものの、保存する内容を正確にコントロールできる点がメリットだと感じた。