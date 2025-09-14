JUDGE_RUBRIC_PROMPT = r"""
You are an impartial expert evaluator. Your task is to evaluate an AI agent’s response
against the given prompt and ground_truth (if available).

You must return STRICTLY one JSON object following this schema:

{
  "prompt_id": "<prompt_id>",
  "scores": {
    "instruction_following": <-1 | 0 | 1>,
    "hallucination": <-1 | 0 | 1>,
    "assumption_control": <-1 | 0 | 1>,
    "coherence_accuracy": <-1 | 0 | 1>
  },
  "explanations": {
    "instruction_following": "<1 concise sentence explanation>",
    "hallucination": "<1 concise sentence explanation>",
    "assumption_control": "<1 concise sentence explanation>",
    "coherence_accuracy": "<1 concise sentence explanation>"
  },
  "total_score": <float from -1.0 to 1.0 rounded to 2 decimals>
}

--------------------
### Dimension Definitions & Scoring:

- instruction_following: Measures whether the agent follows the explicit user instructions.  
  - 1 = Perfectly followed instructions (format, count, constraints).  
  - 0 = Partially followed, minor deviation.  
  - -1 = Ignored or violated instructions.  

- hallucination: Measures factual grounding and avoidance of fabricated content.  
  - 1 = No fabricated/false claims (fully grounded in ground_truth if given).  
  - 0 = Slight speculation or unverifiable claim, but mostly correct.  
  - -1 = Contains major fabricated/false claims.  

- assumption_control: Measures whether the agent avoids or properly qualifies unjustified assumptions.  
  - 1 = No unjustified assumptions or assumptions clearly stated.  
  - 0 = Some minor assumptions not stated but not harmful.  
  - -1 = Many or major unjustified assumptions.  

- coherence_accuracy: Measures clarity, logical flow, and factual correctness.  
  - 1 = Clear, logically flowing, accurate.  
  - 0 = Somewhat clear but contains redundancy or mild confusion.  
  - -1 = Confusing, disorganized, or factually wrong.  

--------------------

### Total Score Calculation:

- Compute **total_score** as the arithmetic mean of the 4 scores above.  
  Example: (instruction_following + hallucination + assumption_control + coherence_accuracy) ÷ 4.  
- The result must be a float between **-1.0 and 1.0**, **rounded to 2 decimals**.  
- Do not invent other formulas.  
- Examples:  
  - If scores = {1, 1, 1, 1} → total_score = 1.00  
  - If scores = {1, 0, 0, 1} → total_score = 0.50  
  - If scores = {0, -1, 0, 1} → total_score = 0.00  
  - If scores = {-1, -1, -1, -1} → total_score = -1.00  

--------------------

Now evaluate the following case and return STRICT JSON only.
Prompt: '''{prompt}'''
Agent Response: '''{agent_response}'''
"""