# AgentCascade — Build-It-Yourself Plan

Work in this folder, on your own machine, with your own terminal/editor, Docker, and Anthropic API key. I'll read whatever you write here (Downloads is already connected in our Cowork session) and give you real review — bugs, design issues, what to think about next — rather than rewriting it for you.

**Ground rule:** `~/Downloads/agentcascade` (the version I already built) is the answer key, not a reference. Attempt each phase for real first — struggle with it, get it wrong, ask me questions — and only diff against that repo *after* you've got your own version working, to see what you did differently and why.

## The four phases

1. **Router + solver tiers** — feature engineering, honest evaluation, cost/accuracy tradeoffs. No Docker or API key needed — start here.
2. **Caching** — a correctness bug lives here if you're not careful; you'll probably find it yourself.
3. **Kafka** — real streaming, producer/consumer, and a design pattern (dependency inversion) that makes streaming code testable without a live broker.
4. **Real LLM tiers** — swap the deterministic solvers for real Claude models, track real token cost.

We'll do one phase at a time. Come back and tell me when you're ready to move on, stuck, or want a design reviewed.

---

## Phase 1 spec — build this yourself, no code from me

**The problem:** given an arithmetic expression as a string, predict — *before* solving it — whether a fast-but-dumb evaluator will get the right answer, or whether it needs to be escalated to a slow-but-correct one. This is the whole "cheap model vs. expensive model" routing problem, just with two deterministic stand-ins instead of two LLMs (no API key needed to learn the core mechanism).

**Build these five pieces, in this order:**

1. **Task generator.** Write a function that produces random arithmetic expression strings with varying structure — some flat (`3 + 4`), some deeply nested with parentheses (`(3 + 4) * (5 - (2 / 1))`). The nesting/parenthesization is what creates genuinely "hard" cases later. Think about: how do you guarantee the expression is always valid (no division by zero, no syntax errors)?

2. **Two solvers.**
   - A **naive** evaluator: reads the expression strictly left to right, applying each operator immediately, with no concept of operator precedence and no parentheses handling. (Question to think about: what's the simplest way to tokenize the expression for this?)
   - An **exact** evaluator: correctly respects precedence and parentheses. (Python gives you a shortcut here — think about what it is, and also think about why using it directly might be a security smell in a real system, even though it's fine for this learning exercise.)

3. **Ground truth labeling.** For a batch of generated expressions, run BOTH solvers and record whether the naive one's answer matches the exact one's. This is your label — did the naive tier actually succeed, computed for real, not assumed.

4. **Feature engineering.** Come up with features you can compute from the expression *text alone*, without solving it — because at decision time, you don't know the answer yet. Think about what structurally makes an expression hard for a naive left-to-right evaluator: what would you look at? (Hint categories, not answers: something about length, something about parentheses, something about which operators are present and where.)

5. **Train + evaluate.**
   - Split your labeled data into train/test. Think about whether a *random* split is fair here or whether it matters (compare to why FraudGuard-Agent used a *time-based* split for fraud data — does that concern apply here too, or not? Why?)
   - Train a classifier (scikit-learn — logistic regression or gradient boosting, your call) to predict the "did naive succeed" label from your features.
   - Evaluate it properly: accuracy alone can lie to you — think about *when* and *why*, and report at least one metric that wouldn't be fooled by a model that just always predicts the majority class.
   - Then compute the number that actually matters: total cost under three strategies — always-naive, always-exact, and routed-by-your-classifier — using a made-up but reasonable cost ratio between the two tiers (e.g. naive costs 1 unit, exact costs 12). Does routing actually beat both baselines on the accuracy/cost tradeoff?

**Concepts worth understanding as you go (look these up, don't just take my word for them):** feature leakage (why using the answer, or something derived from it, in your features would invisibly ruin the whole exercise), class imbalance, precision vs. recall vs. accuracy, and why "the model is 90% accurate" is meaningless without knowing the base rate.

**When you're done, tell me and share the code (or just point me at this folder) and I'll review it** — I'll look for: does your naive evaluator actually fail on the cases it should, are your features actually leak-free, does your evaluation report the right things, and does your cost comparison logic hold up. I won't hand you fixes outright — I'll point at what's wrong and ask you what you think is happening, same as a real code review.
