"""
evaluate_model.py

Runs a set of test conversations through the philosopher bot, then uses
an LLM judge to score each response against a rubric covering:
validity, focus, groundedness, synthesis, progress (multi-turn only),
error recovery, and voice consistency. See eval/rubric.md for full
definitions.

This is an LLM-as-judge setup -- it's a useful automated signal, not a
substitute for reading transcripts yourself. See the "known limitation"
note in eval/rubric.md about judging a model with itself.

Prerequisites:
    - Ollama running with a model pulled (for both the bot and the judge)
    - build_index.py already run (the eval harness reuses your retrieval setup)

Usage:
    python3 evaluate_model.py
    python3 evaluate_model.py --judge-model llama3.1:8b --bot-model llama3.2:3b
    python3 evaluate_model.py --test-file eval/test_cases.jsonl --output-dir eval/results
"""

import os
import re
import json
import argparse
from datetime import datetime

import ollama

from query import load_retrieval, retrieve, SYSTEM_PROMPT, OLLAMA_MODEL as DEFAULT_BOT_MODEL

DEFAULT_TEST_FILE = "eval/test_cases.jsonl"
DEFAULT_OUTPUT_DIR = "eval/results"
DEFAULT_JUDGE_MODEL = DEFAULT_BOT_MODEL  # see rubric.md re: self-judging bias

RUBRIC = {
    "validity": "Is the reasoning logically sound -- free of contradictions, non-sequiturs, or invalid inferences?",
    "focus": "Does the response directly address the question asked, without wandering into tangents?",
    "groundedness": "Are claims consistent with real philosophical positions, not fabricated specifics or misattributed ideas?",
    "synthesis": "Does it speak in one coherent voice, weaving perspectives together -- not mechanically listing 'X says... Y says...'?",
    "progress": "Does each turn build on the conversation so far, rather than repeating itself or ignoring prior context? (Only meaningful for multi-turn conversations.)",
    "error_recovery": "When a question exceeds the bot's grounding, does it acknowledge the gap instead of confabulating a confident but fabricated answer?",
    "voice_consistency": "Does the established persona (a synthesized philosophical thinker) hold up across the conversation, regardless of the user's tone?",
}

JUDGE_PROMPT = """You are evaluating an AI philosophy chatbot's conversation against a rubric. Score each dimension from 1 (poor) to 5 (excellent), with a one-sentence justification for each score. Be honest and critical -- do not default to high scores.

DIMENSIONS:
{dimension_definitions}

CONVERSATION TO EVALUATE:
{transcript}

Respond with ONLY a JSON object in this exact format, nothing else, scoring every dimension listed above:
{{"validity": {{"score": <1-5>, "note": "<one sentence>"}}, "focus": {{"score": <1-5>, "note": "..."}}, "groundedness": {{"score": <1-5>, "note": "..."}}, "synthesis": {{"score": <1-5>, "note": "..."}}, "progress": {{"score": <1-5 or null if single-turn>, "note": "..."}}, "error_recovery": {{"score": <1-5>, "note": "..."}}, "voice_consistency": {{"score": <1-5>, "note": "..."}}}}
"""


def load_test_cases(path):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_conversation(test_case, collection, embed_model, bot_model):
    """Runs a (possibly multi-turn) test case through the RAG pipeline and
    returns the full transcript as a list of {speaker, text} turns."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    transcript = []

    for turn in test_case["turns"]:
        passages = retrieve(collection, embed_model, turn)
        context = "\n\n---\n\n".join(passages)
        user_content = f"BACKGROUND KNOWLEDGE:\n{context}\n\nQUESTION: {turn}"
        messages.append({"role": "user", "content": user_content})

        response = ollama.chat(model=bot_model, messages=messages)
        answer = response["message"]["content"]
        messages.append({"role": "assistant", "content": answer})

        transcript.append({"speaker": "user", "text": turn})
        transcript.append({"speaker": "bot", "text": answer})

    return transcript


def format_transcript(transcript):
    lines = []
    for turn in transcript:
        role = "User" if turn["speaker"] == "user" else "Bot"
        lines.append(f"{role}: {turn['text']}")
    return "\n\n".join(lines)


def extract_json_block(raw_response):
    """Pulls the first {...} block out of a model response, tolerating
    markdown fences or preamble text."""
    cleaned = raw_response.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def judge_conversation(transcript, judge_model, is_multi_turn):
    dimension_definitions = "\n".join(f"- {k}: {v}" for k, v in RUBRIC.items())
    prompt = JUDGE_PROMPT.format(
        dimension_definitions=dimension_definitions,
        transcript=format_transcript(transcript),
    )
    response = ollama.chat(model=judge_model, messages=[{"role": "user", "content": prompt}])
    scores = extract_json_block(response["message"]["content"])

    if scores is None:
        return None

    # Validate shape: every rubric key present with a score/note pair
    for key in RUBRIC:
        if key not in scores or "score" not in scores[key]:
            return None

    if not is_multi_turn:
        scores["progress"]["score"] = None
        scores["progress"]["note"] = "N/A (single-turn test case)"

    return scores


def build_report(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    raw_path = os.path.join(output_dir, f"results_{timestamp}.jsonl")
    with open(raw_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Aggregate averages per dimension across cases where it was scored
    totals = {k: [] for k in RUBRIC}
    for r in results:
        if r["scores"] is None:
            continue
        for k, v in r["scores"].items():
            if v["score"] is not None:
                totals[k].append(v["score"])

    report_lines = [
        "# Evaluation Report",
        f"\nGenerated: {timestamp}",
        f"\nTest cases run: {len(results)}",
        f"Test cases judged successfully: {sum(1 for r in results if r['scores'] is not None)}",
        "\n## Average scores by dimension\n",
        "| Dimension | Average | # Scored |",
        "|---|---|---|",
    ]
    for k in RUBRIC:
        scores = totals[k]
        avg = f"{sum(scores) / len(scores):.2f}" if scores else "N/A"
        report_lines.append(f"| {k} | {avg} | {len(scores)} |")

    report_lines.append("\n## Flagged cases (any dimension scored 1-2)\n")
    flagged = []
    for r in results:
        if r["scores"] is None:
            flagged.append(f"- **{r['id']}**: judge output could not be parsed -- check manually")
            continue
        low = [(k, v["score"], v["note"]) for k, v in r["scores"].items() if v["score"] is not None and v["score"] <= 2]
        if low:
            detail = "; ".join(f"{k}={s} ({note})" for k, s, note in low)
            flagged.append(f"- **{r['id']}**: {detail}")

    report_lines.extend(flagged if flagged else ["None -- no dimension scored 1-2 in this run."])

    report_lines.append("\n## Full results\n")
    for r in results:
        report_lines.append(f"### {r['id']}")
        report_lines.append(f"\n**Transcript:**\n\n```\n{format_transcript(r['transcript'])}\n```\n")
        if r["scores"] is None:
            report_lines.append("_Judge output could not be parsed for this case._\n")
        else:
            for k, v in r["scores"].items():
                score_display = v["score"] if v["score"] is not None else "N/A"
                report_lines.append(f"- **{k}**: {score_display} -- {v['note']}")
        report_lines.append("")

    report_path = os.path.join(output_dir, f"report_{timestamp}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return report_path, raw_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate the philosopher bot against a scoring rubric.")
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE, help=f"Path to test cases JSONL (default: {DEFAULT_TEST_FILE})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"Where to write results (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--bot-model", default=DEFAULT_BOT_MODEL, help=f"Ollama model being tested (default: {DEFAULT_BOT_MODEL})")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help=f"Ollama model used as judge (default: same as --bot-model -- see rubric.md for why a different model is better if you have one)")
    args = parser.parse_args()

    if not os.path.exists(args.test_file):
        print(f"Test file not found: {args.test_file}")
        return

    if args.judge_model == args.bot_model:
        print("[note] Judge model is the same as the bot model. Scores will be a rough")
        print("       signal, not an independent read -- see eval/rubric.md. Pass a")
        print("       different --judge-model if you have another model pulled.\n")

    print("Loading retrieval (embedding model + vector store)...")
    collection, embed_model = load_retrieval()

    test_cases = load_test_cases(args.test_file)
    print(f"Loaded {len(test_cases)} test cases from {args.test_file}\n")

    results = []
    for i, case in enumerate(test_cases):
        print(f"[{i + 1}/{len(test_cases)}] Running '{case['id']}' ({case['type']})...", end=" ", flush=True)

        transcript = run_conversation(case, collection, embed_model, args.bot_model)
        is_multi_turn = case["type"] == "multi_turn"
        scores = judge_conversation(transcript, args.judge_model, is_multi_turn)

        results.append({
            "id": case["id"],
            "type": case["type"],
            "dimension_focus": case.get("dimension_focus", []),
            "transcript": transcript,
            "scores": scores,
        })

        if scores is None:
            print("done (judge output unparseable)")
        else:
            flags = [k for k, v in scores.items() if v["score"] is not None and v["score"] <= 2]
            print(f"done{' [FLAGGED: ' + ', '.join(flags) + ']' if flags else ''}")

    report_path, raw_path = build_report(results, args.output_dir)

    print(f"\nDone. Report: {report_path}")
    print(f"Raw results: {raw_path}")


if __name__ == "__main__":
    main()