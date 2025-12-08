# Core Proofs (On Paper)

This document states and proves several mathematical lemmas that support key STELE metrics and standards.

## 1. OR-Ensemble False-Negative Bound

**Context.**  

Let \(Y \in \{0,1\}\) denote whether a case is Tier-0 harmful (\(Y = 1\)) or not (\(Y = 0\)).  
Let \(J_1, \dots, J_k\) be binary judges. For each \(i\),

\[
J_i = \begin{cases}
1 & \text{judge flags the case as harmful},\\
0 & \text{judge does not flag the case as harmful}.
\end{cases}
\]

Define the false-negative rate of judge \(i\) by

\[
p_i := \Pr[J_i = 0 \mid Y = 1].
\]

Assume that \(J_1, \dots, J_k\) are conditionally independent given \(Y = 1\).

Define the OR-ensemble decision:

\[
J_{\text{OR}} = \begin{cases}
1 & \text{if } \exists i \in \{1,\dots,k\} \text{ with } J_i = 1,\\
0 & \text{otherwise}.
\end{cases}
\]

**Lemma (ENSEMBLE_OR_FN_BOUND).**  
Under conditional independence given \(Y = 1\),

\[
\Pr[J_{\text{OR}} = 0 \mid Y = 1] = \prod_{i=1}^k p_i.
\]

In particular, if \(p_i \le p\) for all \(i\), then

\[
\Pr[J_{\text{OR}} = 0 \mid Y = 1] \le p^k.
\]

**Proof.**

We compute the event \(\{J_{\text{OR}} = 0\}\) given \(Y = 1\).

By definition, \(J_{\text{OR}} = 0\) exactly when all judges output \(0\). So

\[
\{J_{\text{OR}} = 0\} = \bigcap_{i=1}^k \{J_i = 0\}.
\]

Taking probabilities conditioned on \(Y = 1\),

\[
\Pr[J_{\text{OR}} = 0 \mid Y = 1] = \Pr\Big[\bigcap_{i=1}^k \{J_i = 0\} \,\Big|\, Y = 1\Big].
\]

By conditional independence of the \(J_i\) given \(Y = 1\),

\[
\Pr\Big[\bigcap_{i=1}^k \{J_i = 0\} \,\Big|\, Y = 1\Big]
= \prod_{i=1}^k \Pr[J_i = 0 \mid Y = 1]
= \prod_{i=1}^k p_i.
\]

This proves the first statement.

For the second statement, assume \(p_i \le p\) for all \(i\). Then

\[
\prod_{i=1}^k p_i \le \prod_{i=1}^k p = p^k,
\]

since \(p_i \le p\) and all terms are nonnegative. Therefore

\[
\Pr[J_{\text{OR}} = 0 \mid Y = 1] = \prod_{i=1}^k p_i \le p^k.
\]

\(\square\)

---

## 2. Parity Gap Bounds

**Context.**  

Let \(L\) be a set of languages and let \(m : L \to [0,1]\) be a performance metric (for example, Tier-0 recall). Let:

- \(L_0 \subseteq L\) be the set of reference languages (e.g., `L0_ANCHOR`).  
- \(L_1 \subseteq L\) be the set of imperial languages (e.g., `L1_IMPERIAL`).

For two languages \(a, b \in L\), define the pairwise gap

\[
g(a,b) := |m(a) - m(b)|.
\]

Define the L1 parity gap metric as

\[
\text{L1\_PARITY\_GAP} := \max_{a\in L_0,\; b\in L_1} g(a,b).
\]

**Lemma (L1_PARITY_GAP_BOUND).**

1. \(g(a,b) = g(b,a)\) for all \(a,b \in L\).
2. If \(m(a), m(b) \in [0,1]\) for all \(a,b\), then \(g(a,b) \in [0,1]\) and \(\text{L1\_PARITY\_GAP} \in [0,1]\).
3. If there is a fixed reference language \(a_0 \in L_0\) such that
   \(|m(b) - m(a_0)| \le \varepsilon\) for all \(b \in L_1\), then
   \(\text{L1\_PARITY\_GAP} \le \varepsilon\).

**Proof.**

1. Symmetry:

   \[
   g(a,b) = |m(a) - m(b)| = |m(b) - m(a)| = g(b,a),
   \]

   since the absolute value satisfies \(|x| = |-x|\).

2. Range:

   Since \(m(a), m(b) \in [0,1]\), the difference \(m(a) - m(b)\) lies in \([-1,1]\), so
   \(|m(a) - m(b)| \in [0,1]\). Thus \(g(a,b) \in [0,1]\) for all \(a,b\). The maximum of a set
   of values in \([0,1]\) also lies in \([0,1]\), so \(\text{L1\_PARITY\_GAP} \in [0,1]\).

3. Anchor bound:

   Let \(a \in L_0\) and \(b \in L_1\). Then

   \[
   g(a,b) = |m(a) - m(b)| \le |m(a) - m(a_0)| + |m(a_0) - m(b)|
   \]

   by the triangle inequality. If we take \(a = a_0\), then \(|m(a) - m(a_0)| = 0\), and we have

   \[
   g(a_0,b) = |m(a_0) - m(b)| \le \varepsilon.
   \]

   Therefore the maximum of \(g(a,b)\) over \(a\in L_0, b\in L_1\) is at most \(\varepsilon\), so

   \[
   \text{L1\_PARITY\_GAP} \le \varepsilon.
   \]

\(\square\)

---

## 3. Composite Risk Monotonicity

**Context.**  

STELE defines a composite risk score

\[
R_{\text{composite}} = R_{\text{threat}} \cdot R_{\text{domain}} \cdot O_{\text{governance}},
\]

where

- \(R_{\text{threat}} \in [0,1]\) is a threat score,
- \(R_{\text{domain}} \in [0,1]\) is a domain harm score,
- \(O_{\text{governance}} \in [0, M]\) is a governance/overlay weight, with \(M > 0\) fixed (e.g., \(M = 10\)).

**Lemma (COMPOSITE_RISK_MONOTONE).**

For fixed \(R_{\text{domain}}\) and \(O_{\text{governance}}\), \(R_{\text{composite}}\) is non-decreasing in \(R_{\text{threat}}\). Similarly, for fixed \(R_{\text{threat}}\) and \(O_{\text{governance}}\), it is non-decreasing in \(R_{\text{domain}}\). For fixed \(R_{\text{threat}}\) and \(R_{\text{domain}}\), it is non-decreasing in \(O_{\text{governance}}\).

Moreover, if all three inputs are nonnegative and bounded above as above, then \(R_{\text{composite}} \in [0,M]\).

**Proof.**

Fix \(R_{\text{domain}}\) and \(O_{\text{governance}}\). Write

\[
R_{\text{composite}} = C \cdot R_{\text{threat}},
\]

where \(C = R_{\text{domain}} \cdot O_{\text{governance}}\). Since \(R_{\text{domain}} \ge 0\) and \(O_{\text{governance}} \ge 0\), we have \(C \ge 0\). For nonnegative \(C\), the function \(x \mapsto Cx\) is non-decreasing on \([0,\infty)\). Thus \(R_{\text{composite}}\) is non-decreasing in \(R_{\text{threat}}\).

The same argument applies to \(R_{\text{domain}}\) and \(O_{\text{governance}}\) in turn, holding the other two constants.

For the range:

\[
R_{\text{composite}} = R_{\text{threat}} \cdot R_{\text{domain}} \cdot O_{\text{governance}}.
\]

All three factors are nonnegative, so \(R_{\text{composite}} \ge 0\). Also,

\[
R_{\text{threat}} \le 1,\quad R_{\text{domain}} \le 1,\quad O_{\text{governance}} \le M,
\]

so

\[
R_{\text{composite}} \le 1 \cdot 1 \cdot M = M.
\]

Thus \(R_{\text{composite}} \in [0,M]\).

\(\square\)
