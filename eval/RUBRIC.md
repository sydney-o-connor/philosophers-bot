# Evaluation Rubric

`evaluate_model.py` scores the bot's responses against these seven
dimensions, each on a 1-5 scale. This file is the human-readable
reference; the same definitions are embedded in the judge prompt inside
`evaluate_model.py`, so keep them in sync if you edit either.

| Dimension | What it measures | 1 (poor) | 5 (excellent) |
|---|---|---|---|
| **Validity** | Is the reasoning logically sound — free of contradictions, non-sequiturs, or invalid inferences? | Argument doesn't follow from its own premises, or contradicts itself | Each claim follows clearly from what precedes it |
| **Focus** | Does the response directly address the question asked? | Wanders into tangents or answers a different question | Stays tightly on what was actually asked |
| **Groundedness** | Are claims consistent with real philosophical positions, not fabricated specifics or misattributed ideas? | Invents quotes, dates, or positions no philosopher actually held | Everything traceable to real philosophical reasoning or source material |
| **Synthesis** | Does it speak in one coherent voice, weaving perspectives together — not mechanically listing "X says... Y says..."? | Reads like a citation list stapled together | Reads like one mind that has genuinely integrated the material |
| **Progress** *(multi-turn only)* | Does each turn build on the conversation so far, rather than repeating itself or ignoring prior context? | Restates earlier points, ignores follow-up questions | Clearly advances the conversation, responds to what was just said |
| **Error Recovery** | When a question exceeds the bot's grounding (out-of-scope philosopher, made-up topic, trick question), does it acknowledge the gap instead of confabulating? | Invents a confident-sounding but fabricated answer | Honestly flags the limit, or reasons carefully from real principles while saying where it's extrapolating |
| **Voice Consistency** | Does the established persona (synthesized philosophical thinker) hold up across the conversation, regardless of the user's tone? | Breaks character, becomes generic or robotic | Maintains a consistent, thoughtful voice throughout |

## How scoring works

- Each test case in `eval/test_cases.jsonl` tags which dimensions are
  most relevant to it (`dimension_focus`), but the judge scores **all**
  applicable dimensions for every case — `dimension_focus` is mainly for
  your own reference when reading the report, and for filtering.
- `progress` is only scored for multi-turn test cases (single questions
  don't have a "progress" to measure).
- The judge model returns a 1-5 score plus a one-sentence justification
  per dimension. Scores of 1-2 are flagged in the report as needing a
  human look.

## A known limitation: judge bias

If you use the same model as both the bot being tested and the judge,
treat the scores as a rough signal, not ground truth — a model scoring
its own output tends to be more lenient than an independent judge would
be. If you have a second, larger, or otherwise different model pulled in
Ollama, point `--judge-model` at that instead for a more honest read.
Either way, the report is meant to surface things worth reading yourself,
not to replace actually reading the transcripts.