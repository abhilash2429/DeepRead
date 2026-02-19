CODE_GEN_PROMPT = """
Generate PyTorch code only.

Rules:
- Add inline comments for equation-backed lines: # Eq. (N)
- Label assumptions: # ASSUMED: <reason>
- Label inferences: # INFERRED: <reason>
- Never silently invent missing hyperparameters.
- Prefer readability over compactness.
- If a value is missing, leave a TODO-style placeholder and annotate with # ASSUMED or # INFERRED.
- Return only Python code (no markdown fences).
"""
