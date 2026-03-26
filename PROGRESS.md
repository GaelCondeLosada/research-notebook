# Research Notebook — Progress Log

This file tracks progress across all research sessions.

---

## 2026-03-26: Sequence Memory Capacity in Small Networks

**Topic:** `topics/sequence-memory-capacity/`

**Objective:** For a recurrent network of N ≤ 100 neurons that sees a sequence once, derive and experimentally validate how long a sequence L(N) can be perfectly replayed. Deliver: (1) a working Python simulation comparing coding schemes, (2) at least one theoretical capacity bound.

### Key Results

**Theoretical (4 theorems):**
1. Linear pseudoinverse: L* = N + 1 (tight, proven)
2. Feature expansion: L* = M + 1 (hidden dimension M, proven)
3. Fundamental tradeoff: L ≤ P/N + 1 for P fixed parameters (proven)
4. Information capacity: Laplace maximizes entropy per pattern under L1 (metabolic) constraints

**Experimental:**
- Linear: L = N+1 confirmed across all distributions
- Echo state (tanh): L/M ≈ 0.3-0.9, sublinear (tanh saturation)
- Modern Hopfield (normalized): L ≥ 60N for N=50 (search ceiling hit, true capacity much higher)
- Key finding: normalizing Laplace patterns eliminates their Hopfield disadvantage (raw: 18N, normalized: ≥60N)

**Parametric vs Non-parametric dichotomy:**
- Parametric (linear, echo state): capacity bounded by P/N, linear in parameter count
- Non-parametric (Hopfield): exponential capacity 2^Θ(N), but parameter count grows with L

### Files
- `topics/sequence-memory-capacity/2026-03-26-session.tex` — LaTeX report
- `topics/sequence-memory-capacity/simulation.py` — Main experiments (5 experiments)
- `topics/sequence-memory-capacity/sequence_memory.py` — Clean reusable framework (LinearMemory, EchoStateMemory, HopfieldMemory)
- `topics/sequence-memory-capacity/measure_hopfield.py` — Focused Hopfield capacity & coherence measurements
- `topics/sequence-memory-capacity/results.json` — Raw experimental data

### Session Log

**13:49 — Wide Mode (Literature Search)**
- Key papers: one-shot hippocampal model (PLOS 2024), Hopfield Networks is All You Need (Ramsauer 2020), matrix RNN capacity, sparse coding/Laplace
- Switched to Deep Mode at 13:55

**13:55-14:00 — Theory development**
- Proved 4 theorems on capacity bounds
- Key insight: capacity and coding are separable design choices

**14:00-14:10 — First simulation run**
- Linear N+1 confirmed, echo state sublinear, Hopfield hit ceiling

**14:10-14:33 — Extended experiments**
- Normalized Hopfield: massive capacity improvement
- Laplace normalization eliminates disadvantage (surprise finding)
- Coherence analysis: normalized Laplace ≈ Gaussian coherence

**14:33-15:00 — Report writing and probe experiments**
- LaTeX report complete with all results
- Probing true Hopfield capacity at larger L ranges
