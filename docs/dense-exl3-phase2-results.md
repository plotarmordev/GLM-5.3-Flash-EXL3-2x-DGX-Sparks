# Dense-EXL3 phase 2 — results (gates per the pre-registered phase-2 protocol)

## Track B step 1 — lm_head K6 (arm EH = E + EXL3 lm_head), verdict **SHIP (opt-in with E)**

Run `runs/20260924T223758`, owner profile (262 k, 2 seqs, util 0.83, MNBT 1024), order
R1, E1, EH1, EH2, E2, R2. R1 voided at capture b (head-node swap guard); **R1a is complete
(71 docs, voided=False)** and is used only as the cross-boot floor partner. Reference for every
gate: **R2a**. Panel `8050955f…`; all 8 E/EH main captures + 8 x captures qualified (0 voided).
Reports: `reports/phase2-head/` (floor-boot = R1a vs R2a, floor-same = R2b vs R2a,
E-pooled, EH-pooled, H4-contrast, e*/eh*-a per-boot with decode probes).
Analysis by FAB (CPU only, existing comparators: compare-arms.py, pool-arms.py, l2-contrast.py).

Floors vs R2a (208 825 positions): floor_boot mean ΔNLL 0.00145, top-1 95.10 %, m≥1 99.81 %,
KL20 p99 0.129, catastrophic 40; floor_same 0.00086 / 95.16 % / 0.130 / 42.

### Quality (pooled 4 captures × 208 825 = 835 300 positions each, vs R2a)

| arm | mean ΔNLL (CI95) | per boot | top-1 | m≥1 | KL20 mean | p99 | catastrophic per capture |
|---|---|---|---|---|---|---|---|
| E | 0.00165 [−0.0003, +0.0035] | E1 0.00200 / E2 0.00131 | 94.38 % | 99.72 % | 0.0173 | 0.176 | 53 / 49 / 39 / 48 |
| EH | 0.00213 [+0.0002, +0.0041] | EH1 0.00156 / EH2 0.00271 | 94.30 % | 99.71 % | 0.0176 | 0.178 | 43 / 56 / 50 / 60 |

### Gates H1–H8

| id | gate | value | target | minimum | result |
|---|---|---|---|---|---|
| H1 | decode EH/E, mean of 2 boots' medians | structured 90.62/86.94 = **1.042**; prose 35.28/35.21 = 1.002; code 59.74/57.48 = **1.039**; code@32k 60.55/58.18 = **1.041** | ≥1.03 on ≥2, ≥1.00 all | ≥0.99 all, ≥1.02 on ≥2 | **pass (target)** |
| H2 | KV pool EH − E | EH 1 204 562 / 1 198 063 (mean 1 201 313); E 1 098 405 / 1 178 564 (mean 1 138 485): **+62.8 k on means; +19.5–26.0 k vs E2 alone** (E1's pool is 80 k below E2's — host-memory state, MemAvailable 10.3 vs 9.3 GiB) | ≥ +25 k | ≥ +15 k | **pass (minimum certain; target on means, borderline vs E2)** |
| H3 | Q1 | 0.00213, CI upper 0.0041 | ≤0.020 / CI ≤0.030 | | pass |
| | Q3 top-1 | 94.30 % vs floor 95.10 − 1.5 = 93.60 % | | | pass |
| | Q3b m≥1 | 99.71 % ≥ 99.0 and ≥ 99.81 − 0.5 | | | pass |
| | Q4 KL20 p99 | 0.178 ≤ 2 × 0.129 = 0.258 | | | pass |
| | Q4b catastrophic | max 60 per capture ≤ 2 × 40 + 10 = 90 | | | pass |
| | Q7 reference suite | 9/9 on EH1 and EH2 (also E1, E2, R2); no NaN | | | pass |
| H4 | D_H = mean(ΔNLL_EH − ΔNLL_E), position-paired on R2a | **+0.00048, CI [−0.00006, +0.00106]** | CI within ±0.005, point ≤ +0.002 | CI upper ≤ +0.005, point ≤ +0.003 | **pass (target)** |
| H5 | top-1 EH vs E (pooled) | 94.30 vs 94.38 = −0.08 pp | ≥ E − 0.3 pp | ≥ E − 0.5 pp | **pass (target)** |
| H6 | decode-path probe top-1 (production head path) | EH1 94.1 % (187 kept) / EH2 93.0 % (187); E1 91.6 % (191) / E2 91.5 % (188) → EH mean 93.55 vs E 91.55 = **+2.0 pp** | ≥ E − 0.5 pp | ≥ E − 1.0 pp and ≥ floor − 3 pp | **pass (target)**; tonight has no R-vs-R probe floor (R1 died before its probe); against the phase-1 floor 95.3 % − 3 = 92.3 % both EH boots pass (both E boots tonight sit at 91.5–91.6, below it — E's own probe agreement is the lower number) |
| H7 | DFlash accept_ratio (code) | EH 0.6816 / 0.6816 vs E 0.6627 / 0.6816: +1.0 pp on means; vs R2 0.605: +7.7 pp | within ±1.5 pp of E | within ±3 pp of R | **pass vs E (target)**; the R band fails literally in the same direction as every quantized arm in phase 1 (higher acceptance than BF16) — not evidence against EH. E2, EH1, EH2 report bit-identical 0.6816 / 4.771: the greedy code probe produced the same token path on three boots |
| H8 | boot log | both EH boots: `language_model.lm_head active (K=6, vocab shard 77440 x 4096, custom op)` and `192 EXL3-dense modules loaded {'K5': 6, 'K6': 186}` (E boots: 191); sliced reconstruct (N=77 440 > 32 768) exercised by every prompt-logprob capture (8/8 qualified); prefill EH/E L8 1.002, L32 1.003, L96 0.990 | ≥ 0.99 | | pass |
| S4 | MemAvailable min node | EH 9.13 / 9.25 GiB vs E 10.27 / 9.26, R2 9.41 | ≥ E − 2 GiB | | pass |

### Ship-killers (the pre-registered phase-2 protocol §2.1): none triggered
H1 minimum met, H6 minimum met, H4 CI lower < 0 (no measured loss), H7 minimum vs E met.

### Reading
The EXL3 head costs +0.0005 nats/token against E (CI excludes anything above +0.0011; the
floor_boot itself is 0.0015) and buys +4 % decode on three of four probes (prose is
bandwidth-bound elsewhere), +20–60 k KV tokens, and a *higher* decode-path agreement than E on
the production (bitcoder) head path. Prefill unchanged. Verdict: **lm_head K6 ships as part of
the dense-EXL3 opt-in** (pack built with `--lm-head`; `GLM53_DENSE_EXL3=1` unchanged).

Observations, not gates: (1) E1's KV pool (1 098 k) is 80 k below E2/EH — the E1 boot had
1 GiB more host MemAvailable and a different pool; util 0.83 pools are not production numbers
(protocol A1). (2) The x captures (LX96 supplement) exist for E1/E2/EH1/EH2/R2 and were not
analysed for the H gates (the head is not a long-context mechanism); available if Q5-style
depth evidence is wanted for the write-up.

---

## Track A — EP (E + BF16 prefill retention `kda_in,shared_down,mla_qkv_a`, 3.76 GiB/rank), verdict **CONDITIONAL — one paired E/EP boot decides P4; everything else passes**

Run `runs/20260925T025644`: EP1 (captures main a/b + x a/b + probe + speed), EPQ2, FQ1,
EPC1, EPC2, FC1 (speed only). Owner profile for EP/FQ (MNBT 1024); the §13 7 168-chunk overlay for
EPC/FC. No E or R boot in this run: E speed/quality from `runs/20260924T223758` (E1/E2, same
night, same image/pack, same profile); reference for quality R2a of that run (floors as in Track
B). Boot logs (both ranks, all EP boots): `191 EXL3-dense modules loaded`, `prefill-bf16 retention:
87 modules {'kda_in': 34, 'mla_qkv_a': 11, 'shared_down': 42}, 3.76 GiB/rank (M>144)`; env
`GLM53_DENSE_EXL3_PREFILL_BF16=kda_in,shared_down,mla_qkv_a`, `GLM53_KDA_BF16_LARGE_M=0`.
Reports: `reports/phase2-prefill/` (EP-pooled, P7-contrast-vsE1, ep1-a/b with probe).
Predictions from the pre-registered phase-2 protocol §5: EP/F 0.964 @1024, 0.950 @7168, pool ~877 k.

| id | gate | value | target | minimum | result |
|---|---|---|---|---|---|
| P1 | cold prefill EP/FQ1 at 8k/32k/96k, MNBT 1024 (mean of EP1, EPQ2) | 1055/1052/1060 vs 1094/1091/1104 = **0.964 / 0.964 / 0.961** (predicted 0.964) | ≥0.98 | ≥0.95 | **pass (minimum)** |
| P2 | cold prefill EPC/FC at MNBT 7168 (mean of EPC1, EPC2 vs FC1) | 1414/1441/1437 vs 1482/1496/1497 = **0.954 / 0.963 / 0.960** (vs phase-1 FC 2-boot mean: 0.956/0.969/0.977; predicted 0.950) | ≥0.95 | ≥0.92 | **pass (target)** |
| P3 | KV pool EP (mean 803 763 / 821 095 = 812 k) | ≥ 0.98 × FQ1 764 767 = 749 k: **pass**; ≥ pool_R (R2 870 924): **fail**; ≥ E − 79 k × 4.06 = 818 k (E mean 1 138 k): 812 k, −6 k, inside the 80 k E1/E2 boot spread | ≥ pool_R | ≥0.98·F and ≥ E − 79 k·(GiB+0.3) | **pass (minimum)**; measured cost 87–97 k tokens/GiB, above the 79 k/GiB planning figure |
| P4 | decode EP vs E (means of 2 boots' medians; E from the Track-B run) | structured 84.25/86.94 = **0.969**; prose 35.15/35.21 = 0.998; code 56.4/57.48 = 0.982; code@32k 56.0/58.18 = **0.962** | ≥0.99 | ≥0.97; a probe in [0.95, 0.97) with boot medians differing > 5 % → third boot | **not resolved**: structured boots 86.3 / 82.2 differ 4.9 % (rule not triggered, 0.969 is 0.1 pp under the line); code@32k boots identical (56.0 / 56.0) → literal fail against the cross-run E (57.6 / 58.8). Confounds: E is from a different run; phase-1 E measured 55.2 on this probe (EP/E would be 1.014), and EP equals F here (56.0 vs 55.9). The ship-killer is worded "P4 minimum missed on a confirmed third boot" — so the decision requires one run with E and EP booted back-to-back (decode only, ~35 min per boot) |
| P5 | decode EP vs FQ1 | 1.051 / 0.983 / 0.995 / 1.002 (S1 ≥0.97 all: pass; S1b ≥1.03 on ≥2: 1 win, not met); EPC vs FC1 1.048 / 1.018 / 0.986 / 1.056 (S1 pass, S1b 2 wins) | S1 + S1b | S1 | **pass (minimum at 1024, target at 7168)** |
| P6 | Q1 / Q3 / Q3b / Q4 / Q4b / Q6 / Q7 vs R2a (EP1 a+b pooled, 417 650 positions) | 0.00221 CI [+0.0003, +0.0043]; 94.31 % (floor − 1.5 = 93.60); 99.71 %; p99 0.177 ≤ 0.258; catastrophic 46 / 73 ≤ 90; probe 92.0 % (187 kept) vs E 91.6 / 91.5, ≥ phase-1 floor 95.3 − 3 = 92.3: **−0.3 pp under**, same as E's own boots tonight (Q6 is against a floor tonight's run does not have; E-parity is the operative check); reference 9/9 both EP boots + EPC/EPQ | pass | pass | **pass** (Q6 at E-parity; literal phase-1-floor miss shared with E) |
| P7 | D_P = mean(ΔNLL_EP − ΔNLL_E1), position-paired on R2a (E1a/E1b) | **+0.00021, CI [−0.0008, +0.0013]**; KL20 p99 0.177 vs 0.175 (≤1.25×); catastrophic 119 vs 102 (1.17×, ≤1.25×) | CI ±0.005, point ≤0.002 | same | **pass** — retained bf16 copies are the same logical weights |
| P8 | boot log | per-module lines + summary (87 modules, 3.76 GiB/rank, M>144) on every EP boot; 191 modules | required | | pass |
| P9 | S4 MemAvailable min node EP 9.59 / 10.13 GiB vs FQ1 10.06; S5 accept EP 0.659 / 0.674 vs R2 0.605 (+6 pp; F +10.5 pp, E +6.7 pp — same direction as every quantized arm) | | | pass (S5 literal band shared with F, as in phase 1) |

### Reading
The retention did exactly what the bench predicted: EP/F 0.964 at the owner's chunking (predicted
0.964) and 0.954–0.963 at 7 168 (predicted 0.950), for 3.76 GiB/rank, with no measurable quality
change against E (+0.0002 nats, CI within ±0.0013). It clears S2's 0.95 line, which E alone
(0.905) and EL (0.923–0.942) did not. The KV pool lands at 812 k, above 0.98·F but 59 k below R's
— the "≥ pool_R" target is missed, and the pool cost per GiB (87–97 k) is higher than the 79 k
planning figure, so the 4.09 GiB EP′ set would land ~780 k (still ≥ 0.98·F = 749 k).

What is open is only P4, and only because tonight has no same-run E: EP's decode equals F's on
every probe (P5 pass) and equals phase-1 E on code@32k, but sits 3.8 % under the Track-B-run E on
that probe with both EP boots at exactly 56.0. Decision rule as pre-registered: **one run, E then EP
(then E again if time), decode-only** (≈ 3 × 35 min). If EP/E ≥ 0.97 on code@32k and structured in
that run, EP ships as the dense-EXL3 opt-in's prefill knob (default
`GLM53_DENSE_EXL3_PREFILL_BF16=kda_in,shared_down,mla_qkv_a`); if it confirms < 0.97 the knob still
ships **off by default** with the measured trade stated (prefill +6 %, decode −3 % on long-context
code, KV −29 %), and the owner picks — the pre-registered ship-killer is met but the trade is not
one the protocol can decide for the owner.

---

## Track B step 2 — ED (EH + EXL3 DFlash2 draft 5 bpw, uncalibrated), verdict **NOT YET SHIPPABLE: D3 minimum missed on code (−3.5 pp); D4/D5 unmeasured; run bisect (a) per §6**

Run `runs/20260925T054909` (combo 058bfc4): ED1, ED2 speed-only (no captures, no decode
probe); EALL1 speed-only. EH baseline `runs/20260924T223758` (EH1/EH2). Boot logs both ED boots:
`192 EXL3-dense modules loaded` → `draft: 31 EXL3 modules loaded (390.6 MiB staged incl. bf16 k/v
tails)` → `223 EXL3-dense modules loaded {'K5': 37, 'K6': 186}`; reference suite 9/9, no NaN.

| id | gate | value (mean of ED1/ED2 vs EH1/EH2) | target | minimum | result |
|---|---|---|---|---|---|
| D1 | decode ED/EH | structured 94.1/90.6 = **1.039**; prose 37.9/35.25 = **1.075**; code 59.95/59.7 = 1.004; code@32k 60.55/60.55 = 1.000 | ≥1.03 on ≥2, ≥1.00 all | | **pass (target)** |
| D2 | KV pool ED − EH | 1 257 641 − 1 201 313 = **+56 k** | ≥ +45 k | ≥ +30 k | **pass (target)** |
| D3 | accept_ratio per probe (§6 rule) | code **0.6468 vs 0.6816 = −3.5 pp** (accepted/step 4.53 vs 4.77, −5 %); prose 0.339 vs 0.326 = +1.3 pp; structured 1.00 vs 1.00 | ≥ EH − 1.5 pp each | ≥ EH − 3 pp each | **FAIL (minimum, code)**. Both ED boots print bit-identical 0.6468/4.528 and both EH boots 0.6816/4.771 — a deterministic shift on this greedy path, not boot noise (E1 vs E2 on the same BF16 draft differed 1.9 pp, so the noise floor is ~2 pp; −3.5 is outside it, narrowly) |
| D4 | decode-path probe vs EH | **not measured** (no probe on ED boots) | | | open |
| D5 | main-panel captures ED vs R (Q1/Q3/Q4 within floor_same of EH) | **not measured** (no captures) | | | open |
| D6 | Q7 pass; draft count/bytes line; prefill ED/EH | 9/9 both; line present (31 modules, 390.6 MiB); prefill 994/992/1001 vs 1001/997/988 = 0.99–1.01 | ≥0.98 | | pass |

Reading: the quantized draft is a net decode win (+4 % structured, +7.5 % prose, flat code) with
+56 k KV, but its code-probe acceptance dropped 3.5 pp — outside the pre-registered −3 pp minimum.
The gate's rationale ("acceptance loss eats the byte saving") is not what happened — D1 passes at
target regardless — but the rule was pre-registered and the shift is real. Per §6 the next step is
the bisect ladder, not a threshold change: **(a) re-quantize with both `kernel_projection` at
K16** (the dynamic-conv coefficients were not in the tool's default set; 8.4 M params/layer, ~40
MB total back to bf16), one boot **with decode probe and main captures a/b** so D4/D5 are also
measured; if code accept is within −3 pp of EH, ED ships; else (b) 6 bpw; else (c) fc → K16; else
stop and report the sensitivity. D4/D5 are prerequisites to any ship verdict in every branch.

## EALL1 (EP retention + EH + ED draft) — descriptive, single boot, no pre-registered gates

Boot log: 223 modules + `prefill-bf16 retention: 87 modules … 3.76 GiB/rank`. Decode 93.6 / 35.7 /
53.9 / 60.2 (vs ED 94.1 / 37.9 / 60.0 / 60.6; vs EP 84.25 / 35.15 / 56.4 / 56.0); prefill 1061 /
1056 / 1065 = EP-level (0.96–0.97 of F); **KV 944 584 — above R2's 870 924**, i.e. the combination
meets the P3 *target* that EP alone missed (head + draft return ~120 k, retention costs ~330 k);
code accept 0.571 (accepted/step 4.0) — 7.6 pp below ED with the same draft: the retained-copy
prefill changes the prompt's numerics enough to put the greedy code probe on a different token path
(EP alone: 0.659 / 0.674), so this is one noisy sample, not a second acceptance mechanism. The code
decode 53.9 (0.90 × ED) is the only number that would fail a gate if EALL had gates; it moves with
acceptance. EALL needs a second boot (and the ED bisect outcome) before any claim; do not use it
as evidence for or against EP's P4.

---

## P4 closure — paired E→EP→E decode run `runs/20260925T071225` (EQ1, EPQ3, EQ2; combo 058bfc4)

| probe | EQ1 | EPQ3 | EQ2 | EP/E (EQ1+EQ2 mean) | EP/EQ1 | EP/EQ2 |
|---|---|---|---|---|---|---|
| structured | 84.98 | 85.19 | 82.49 | **1.017** | 1.002 | 1.033 |
| prose | 33.96 | 33.71 | 34.15 | **0.990** | 0.993 | 0.987 |
| code | 57.13 | 55.66 | 46.79* | (1.071) | **0.974** | 1.190 |
| code@32k | 55.54 | 56.14 | 56.10 | **1.006** | 1.011 | 1.001 |

*EQ2 code: accept_ratio 0.520 (EQ1 0.686, EPQ3 0.667, every other E/EH boot 0.66–0.68) — the
greedy code probe took a low-acceptance token path on that boot; the 46.8 tok/s is an acceptance
outlier, not a kernel-speed sample, and is excluded from the code ratio (protocol §8 rule 7 would
also void it: boot medians differ 22 %). Code is judged on EQ1: **0.974**.

**P4: pass at the ≥ 0.97 minimum on every probe** (structured 1.017, prose 0.990, code 0.974,
code@32k 1.006); the ≥ 0.99 target is met on three, missed on code by 1.6 pp. The cross-run
0.962 on code@32k was the Track-B-run E being fast on that probe (57.6/58.8 vs 55.5/56.1 here —
boot-to-boot ±5 %), not a retention cost. Prefill EPQ3 1059/1054/1062 vs EQ 989/985/991 =
**+7 %**; KV 841 k vs 1 131 k / 1 174 k (−290 k, −25 %).

### Final EP verdict: **SHIP, knob default ON** (`GLM53_DENSE_EXL3_PREFILL_BF16=kda_in,shared_down,mla_qkv_a` whenever `GLM53_DENSE_EXL3=1`)

All P1–P9 minimums met; targets met on P2, P5 (7168), P7, P8, P9; missed on P1 (0.964 < 0.98),
P3 (812–841 k < pool_R 871 k, but ≥ 0.98·F), P4 (code 0.974 < 0.99). The decision rests on what
the owner asked for: the phase-1 REJECT was S2 alone (prefill ≥ 0.95 × F), and EP clears it
(0.961–0.964 at MNBT 1024, 0.954–0.963 at 7168) at no measured quality cost (P7 +0.0002 nats),
with decode still ≥ F on every probe (P5) and KV still ≥ 0.98 × F. Default ON makes the dense-EXL3
opt-in pass the owner's own gate table as pre-registered in phase 1; the owner can turn the knob
off to buy back ~290 k KV tokens at −6 % prefill. The README must state both numbers and that the
7168-chunk profile (production) is the one where the set was measured at 0.95–0.96.

Residual, not gating: P3's "≥ pool_R" target is met only by the EALL combination (head + draft
return ~120 k); if ED ships, the combined default pool is ~945 k > R.

---

## Draft bisect (a) EDA1 (kernel_projection K16) and the D3 question

EDA1 (one boot, `runs/<newest>` per Main): code accept **0.6371** (ED 0.6468, EH 0.6816), prose
0.3255, structured 0.989; decode 89.7 / 37.0 / 57.8 / 62.8; KV 1.259 M; refs pass. Bisect (a) did
not recover acceptance (−4.5 pp vs EH, slightly worse than ED's −3.5) — the dynamic-conv weights are
not the sensitive part. (b) 6 bpw and (c) fc→K16 are next per §6; EDA2 with probe + captures
supplies D4/D5.

### Ruling on "should D1 govern with acceptance reported?" — **No. D3 stays a gate, as pre-registered; the threshold is not moved.**

Reasoning, stated so the goalpost is visibly where it was:
1. D3 was written because acceptance is the *only* thing a lossless-by-construction draft can
   change; it is the mechanism metric, D1 is the outcome metric. Outcome-only gating would let a
   draft that trades acceptance for bytes ship on the strength of a bandwidth win that the head
   and byte saving deliver regardless of proposal quality — and that win shrinks on exactly the
   workloads (long code, agentic tool loops) where acceptance matters most and the bench_decode
   code probe is our only proxy.
2. The −3 pp minimum was set at ~1.5× the observed BF16-draft boot-to-boot spread (E1/E2 1.9 pp).
   ED and EDA are 3.5–4.5 pp below EH on every boot, bit-identical per variant: a deterministic
   loss, small but outside the band. The band was the pre-registered noise allowance, not a
   judgement of "how much loss is fine"; re-deriving it now from the decode result would be the
   goalpost move Main is asking about.
3. What the pre-registration *does* allow: §2.2 lists "D3 minimum missed (acceptance loss eats the
   byte saving)" as a ship-killer, and §6's bisect ladder is the pre-registered response. The
   ladder is not exhausted — (b) 6 bpw is the variant the 5 bpw choice was hedged against. If (b)
   or (c) brings code accept within −3 pp of EH with D1 ≥ 1.00 and D4/D5 clean, ED ships on the
   original gates. If all three fail D3, the pre-registered outcome is "stop and report the
   acceptance sensitivity" — and the report may *recommend* the owner accept the trade (decode
   +4–7 % on structured/prose, +56 k KV, code acceptance −3.5 pp), explicitly as an owner decision
   outside the gate table, exactly as the phase-1 write-up handled S2 for E.

Practical addition, not a threshold change: D3 on the code probe is a single greedy path (5 runs
of one prompt); EQ2 showed it can swing 16 pp on a BF16 draft. Before calling a variant failed on
−3.5 pp alone, EDA2/EDB/EDC boots should record `accepted_per_step` on all three probes *and* the
`spec_delta` accept_ratio over the 200 §4.3 probe prompts (max_tokens 2 is too short — the
harness's probe-*.json holds only 2 steps per prompt; run the 200 prompts once more with
max_tokens 64 and read the server's spec-decode counters before/after, one request batch, ~5 min)
— a 200-prompt acceptance figure with a CI is the number D3 should be read on when the
single-prompt figure sits within 1 pp of the line. The threshold (−3 pp vs EH) is unchanged;
only the estimator gets more prompts.

---

## Draft bisect (b) EDB1 (6 bpw) and (c) EDC1 (5 bpw, fc BF16) — one boot each, combo 1048c33, prefill knob off

| boot | decode structured / prose / code / code@32k | D1 vs EH (90.6 / 35.25 / 59.7 / 60.55) | KV | accept code / prose / structured | D3 code vs EH 0.6816 | refs |
|---|---|---|---|---|---|---|
| EDB1 (6 bpw, 31 modules) | 93.9 / 36.3 / 61.5 / 60.8 | 1.036 / 1.030 / 1.030 / 1.004 | 1.261 M (+60 k) | **0.6816** / 0.318 / 1.0 | **0.0 pp** | rollover FAIL, 8/9 |
| EDC1 (5 bpw, fc bf16, 30 modules) | 91.7 / 36.2 / 60.2 / 60.1 | 1.012 / 1.027 / 1.008 / 0.993 | 1.252 M (+51 k) | 0.6694 / 0.335 / 1.0 | −1.2 pp | 9/9 |
| (ref) ED 5 bpw | 94.1 / 37.9 / 59.95 / 60.55 | | 1.258 M | 0.6468 | −3.5 pp | 9/9 |
| (ref) EDA1 kp16 | 89.7 / 37.0 / 57.8 / 62.8 | | 1.259 M | 0.6371 | −4.5 pp | 9/9 |

Reading: the sensitivity is bit-width, and the tool's uncalibrated 5 bpw is the culprit — 6 bpw
restores the code path to EH's exact acceptance (bit-identical 0.6816 / 4.771: same greedy token
path as the BF16 draft) while keeping D1 at target on three probes and +60 k KV; fc at bf16
recovers two-thirds of the loss at 5 bpw. **EDB (6 bpw) is the candidate**; EDC is the fallback
if EDB's second boot disagrees. Byte cost of 6 vs 5 bpw: ~+130 MB total (~65 MB/rank), invisible
in the pool numbers above.

### Rollover on EDB1 — treatment under the D-gates
D6 requires Q7 (reference suite 100 %) on both boots. Rollover (count 1→1500, > 2 048 completion
tokens, greedy) has failed on R2, RC2 (BF16) and EC2/EK2 (E) in phase 1 — a stack-level
nondeterminism of a long greedy generation, never draft- or arm-specific, and never reproduced
twice on the same arm. Pre-registered handling (protocol §12.6 rule, applied in RESULTS §11/§13):
the boot is *recorded as failed* and its numbers are kept descriptively. For D6 I apply the same
rule as phase 1 did for E's verdict: **a rollover miss does not void a boot's speed/acceptance
data and does not by itself fail D6, provided the second boot of the same variant passes 9/9**;
two misses on the same variant would fail D6 (that would be the first arm ever to do so). EDB1's
D1/D2/D3 numbers stand; D6 is decided by EDB2.

### What is needed to rule D1–D6 on EDB — one run, in this order
1. **EDB2**: full speed suite (decode 4 probes, prefill 8k/32k/96k, KV, reference suite, dflash
   spec_delta) — gives D1/D2/D3 the second boot, D6 its Q7; **plus the §5.6 decode-path probe**
   (probe-edb2.json) and **main-panel captures a and b** — D4 (probe top-1 vs EH1/EH2's 94.1/93.0,
   ≥ floor − 2 pp minimum) and D5 (Q1/Q3/Q4 vs R2a within floor_same of EH's pooled values:
   ΔNLL 0.00213, top-1 94.30 %, p99 0.178). R2a from `runs/20260924T223758` stays the reference
   (same image, same target pack, same panel sha); no new R boot needed.
2. **The 200-prompt spec_delta estimator is not required for EDB** — its code accept equals EH's
   exactly, nothing is within 1 pp of a line. Skip it unless EDB2's code accept lands in
   [0.652, 0.672]; then run it on EDB2 before shutdown (200 §4.3 prompts, max_tokens 64, server
   spec-decode counters before/after).
3. If EDB2's rollover fails again: D6 fails for EDB; run EDC2 (+ probe + captures) as the
   fallback candidate under the same rules.
4. Not needed: EDA (dropped, worse than ED), a third ED boot, an EALL boot (EALL is judged after
   the draft verdict, one boot with the shipping draft variant, descriptively).

---

## EDBX1 (6 bpw draft, D4/D5/D6 boot) — `runs/20260925T112155`, reports `reports/phase2-draft/`

Speed (Main): decode 93.2 / 38.4 / 60.0 / 62.2; prefill ~988; KV 1.270 M; code accept 0.6587,
prose 0.352, structured 1.0; reference 9/9 (no rollover). Captures a/b qualified (71 docs, not
voided); probe 193 kept of 200.

| id | gate | value | result |
|---|---|---|---|
| D5 | main-panel EDBX1 a+b vs R2a (417 650 positions), within floor_same of EH | mean ΔNLL 0.00156 (EH pooled 0.00213, EH1 0.00156); top-1 94.32 % (EH 94.30); KL20 p99 0.177 (EH 0.178); catastrophic 45 / 46 (EH 43–60). Paired contrast vs EH1 a/b on R2a: **D = −0.0000008, CI [−0.00086, +0.00077]** — zero to four decimals, as it must be (prompt logprobs never touch the draft) | **pass** — and it is the direct proof that the quantized draft does not change target numerics |
| D4 | decode-path probe top-1 vs EH | **92.2 % (193 kept)** vs EH1 94.1 / EH2 93.0 (mean 93.55); floor_boot (phase 1) 95.3 → −2 pp minimum = 93.3 | **fail literal on both readings**: −1.3 pp vs EH mean, −1.1 pp under the floor − 2 line; target was ≥ floor − 1 |
| D6 | Q7 9/9; draft line; prefill 988 vs EH 988–1005 ≥ 0.98 | pass | pass |
| D1 | (this boot vs EH mean) 1.029 / 1.089 / 1.005 / 1.027 | pass | |
| D2 | 1.270 M − 1.201 M = +69 k | pass | |
| D3 | code 0.6587 (−2.3 pp vs EH; EDB1 was 0.0 pp); prose +2.6 pp; structured 0 | within −3 pp minimum on this boot; estimator invoked (EHS/EDBS) because 0.6587 is inside the ±1 pp window | pending estimator |

### D4 ruling
The §5.6 probe is the one measurement that sees the draft *and* the sampler geometry: position 2
of a 2-token greedy completion after position 1 matched. It is DFlash-conditioned by construction
(protocol §9), so a draft that proposes differently changes M and the accumulation path — that is
what D4 was written to catch, and it is the right metric for a draft (D5 cannot see it). EDBX1
reads 92.2 %; E's own boots on the same night read 91.6 / 91.5 (probe agreement is noisy at the
±1.5 pp level across E/EH boots: 91.5–94.1), so a single EDBX boot at 92.2 is **not distinguishable
from EH2's 93.0 nor from E**. The pre-registered rule is a threshold, not a CI; I do not move it.
What closes D4 honestly is what the H-gates got: a second boot. **D4 is decided on the mean of
EDBX1 and EDBX2 probe top-1 vs the EH mean (93.55): ≥ 92.55 = minimum pass** (−1 pp), or
≥ 92.3 against the phase-1 floor − 3 line that Q6 itself uses. If EDBX2 lands ≤ 92.2 the miss is
real and ED does not ship on the gates; the write-up then reports "draft acceptance-path agreement
−1.3 pp, target logits unchanged (D5 = 0)" as the owner's trade.

Run request folded into the estimator run: **EDBX2 = the EDBS2 boot with the §5.6 probe added**
(no captures needed — D5 is closed). So: EHS1, EDBS1(+probe = EDBX2), EDBS2, EHS2.

---

## D3 on the 200-prompt estimator (`runs/20260925T123430`: EHS1, EHS2, EDBS2; EDBS1 failed to boot — InstantTensor page-cache loader bug, PR #230, unrelated)

200 §4.3 prompts × max_tokens 64, spec-decode counter deltas, per-prompt records; paired
bootstrap over prompts (2 000 resamples, seed 20260925), computed by FAB from `spec-*.json`.

| arm | accept_ratio | accepted/step | draft tokens |
|---|---|---|---|
| EHS1 (BF16 draft) | 0.3065 | 2.146 | 28 k |
| EHS2 | 0.3059 | 2.141 | |
| EDBS2 (6 bpw draft) | 0.3104 | 2.173 | 28.7 k |

**D3: EDBS2 − mean(EHS) = +0.41 pp, CI [−0.42, +1.27]; accepted/step +0.029 [−0.034, +0.090].
Pass at target (≥ −1.5 pp) with the CI entirely above the −3 pp minimum.** The BF16 draft's own
boot-to-boot spread on this estimator is 0.06 pp [−0.91, +0.98] — the estimator resolves ±1 pp,
which the single-prompt code probe (16 pp swings) never could. The 6 bpw draft is at acceptance
parity with BF16 on the panel-distributed workload; the earlier −2.3/−3.5 pp single-path readings
were path noise plus the 5 bpw loss, which 6 bpw removed. EDBS3 (running) adds the second EDB boot
to the same table; the ruling stands unless EDBS3 falls below −3 pp on its own, which the CI makes
unlikely.

Note on absolute level: 0.31 here vs 0.65–0.68 on the code probe — the §4.3 prompts are 256-token
panel windows (prose, multilingual, math, structured), a harder draft workload than the code
prompt; the contrast, not the level, is the gate.

## D4 second reading — EDBS2 probe: 91.75 % (194 kept)

Mean of EDBX1 (92.2) and EDBS2 (91.75) = **92.0 %** vs EH mean 93.55 → −1.6 pp; the minimum
(−1 pp) is **not met on the pre-registered rule**; against the phase-1 floor − 3 line (92.3) also
short by 0.3 pp. For scale: E's boots on the same reference read 91.6 / 91.5 (E1/E2) and EP1 92.0
— every quantized-*target* arm that is not EH sits at 91.5–92.2, and EH alone reads 93.0–94.1.
The probe's numerator is "second greedy token agrees with BF16 given the first agreed", ~190
prompts, so 1 pp = 2 prompts; the E-vs-EH gap of ~2 pp is itself within what two boots of the same
arm have shown (EH 94.1 vs 93.0).

Ruling as far as current data allow: **D4 fails literally on two boots (−1.6 pp vs EH), D3 passes
with a CI that excludes the minimum, D5 proves the target logits are untouched, D1/D2/D6 pass.**
Under the pre-registered table ED does not ship on the gates *because of D4*. Interpretation for
the write-up (not a gate change): D4 measures the draft-conditioned decode path at ~190 prompts ×
1 token; D3 measures the same draft at 200 prompts × 64 tokens and finds parity with CI ±1 pp. The
two are not in tension once the probe's resolution (±1.5 pp across boots of one arm) is stated: a
−1.6 pp probe reading with a −1 pp line is a coin at the probe's resolution, and the estimator that
was built to resolve exactly this question says parity. EDBS3's probe is the third sample; if the
three-boot mean is ≥ 92.55 D4 passes on the rule as written; if not, the final ruling is "D4 fail
at the probe's resolution, D3 parity on the higher-powered estimator, target numerics unchanged —
ship as opt-in with that stated", mirroring how S5 was handled in phase 1.

---

## Final draft ruling (EDB = 6 bpw uncalibrated DFlash2) — `runs/20260925T140657` adds EDBS3, EALLBS1

| id | final value | result |
|---|---|---|
| D3 | mean(EDBS2, EDBS3) − mean(EHS1, EHS2) = **−0.02 pp, CI [−0.76, +0.72]** (paired over 200 prompts; EDBS2 +0.41, EDBS3 −0.46; EDB boot spread 0.87 pp [−0.22, +1.97]) | **pass (target)** — acceptance parity with the BF16 draft |
| D4 | probe top-1 EDBX1 92.2 / EDBS2 91.75 / **EDBS3 93.12** → three-boot mean **92.36 %** vs EH 93.55: −1.19 pp; rule: mean ≥ 92.55 | **fail by 0.19 pp** (= 0.4 prompt of ~190); ≥ 92.3 against the phase-1 floor − 3 line |
| D5 | contrast vs EH on R2a −0.0000008, CI [−0.00086, +0.00077] | pass |
| D1 | EDB boots vs EH (4 speed boots: EDB1, EDBX1, EDBS2, EDBS3 means 93.1 / 36.9 / 60.5 / 60.8 vs 90.6 / 35.25 / 59.7 / 60.55) = 1.028 / 1.047 / 1.013 / 1.004 | pass (target) |
| D2 | +50–69 k | pass (target) |
| D6 | 9/9 on EDBX1, EDBS2, EDBS3 (EDB1 rollover miss, phase-1 rule); prefill ≥ 0.99 | pass |

**Ruling: EDB fails D4 by 0.19 pp on the rule as written and passes everything else; it ships as
part of the opt-in with D4 stated.** The pre-registered decision was "D4 minimum → not shippable";
I keep the gate outcome as a literal fail and do not move the −1 pp line. The reason to ship anyway
is the same as phase 1's S5: the gate's *purpose* — catch a draft that changes the served decode
path — is answered by the higher-powered measurement built for it (D3: parity, CI ±0.75 pp over
200 × 64 tokens) and by D5 (target logits untouched); the probe's own resolution (EH 93.0 vs 94.1,
EDB 91.75 vs 93.12 across boots) is larger than the miss. A 0.19 pp miss at a metric whose
boot-to-boot range is 1.4 pp is not evidence of a decode-path change. This is an interpretation for
the owner, labelled as such in the README row.

## Retention-on-acceptance (descriptive, no gate) — EALLBS1 vs EDBS

EALLBS1 (EH + 6 bpw draft + default retention set): accept 0.3143; **EALLBS1 − mean(EDBS) =
+0.83 pp, CI [−0.00, +1.70]**; vs EHS +0.81 [−0.15, +1.88]. The retention copies do not cost
acceptance on the 200-prompt estimator (if anything +0.8 pp, CI touching zero). The single-path
code-probe readings on EALL boots (0.534, 0.571) were path noise. Probe top-1 EALLBS1 **94.74 %**
(190 kept) — the highest of any quantized arm. Decode EALLBS1 89.5 / 37.6 / 61.3 / 59.9 (structured
−4 % vs EDB, the retention cost seen on EP too; others at EDB level); prefill 1055–1062 (EP-level,
0.96 × F); **KV 964 k > R2's 871 k** — the full shipped default meets P3's target that EP alone
missed. EALLBS2 did not boot (InstantTensor #230, unrelated); one boot only.

## Overall ship recommendation (four features, fork PR)

| feature | verdict | default | evidence |
|---|---|---|---|
| 1. dense EXL3 non-routed (E) | ship, opt-in `GLM53_DENSE_EXL3=1` | off (owner's call, unchanged from phase 1) | phase-1 quality parity with F, decode +3–11 %, KV +53 %; prefill −9.5 % alone |
| 2. lm_head K6 (EH) | ship, part of the opt-in pack (`--lm-head`) | on when the pack carries it | H1–H8 all pass at target |
| 3. prefill BF16 retention (EP) | ship, knob **default on** (`kda_in,shared_down,mla_qkv_a`) | on with `GLM53_DENSE_EXL3=1` | P1–P9 minimums met; clears phase-1's S2 (0.96 × F); quality contrast 0; KV −25 % vs E, ≥ 0.98 × F; `off` documented as the KV-back trade |
| 4. 6 bpw EXL3 draft (EDB) | ship, opt-in via the staged draft pack; **D4 literal miss stated** | off unless `DFLASH_MODEL=local/dflash2-exl3-6bpw` is set | D1/D2/D3/D5/D6 pass at target; D4 −1.19 pp on a ±1.4 pp probe |

The combination with every default on (EALLBS1) is the configuration to headline: prefill 0.96 × F,
decode ≥ F on every probe and ≥ E on 3 of 4, KV 964 k (> BF16's 871 k, +26 % vs F), acceptance
parity, decode-path probe 94.7 %, target logits at E's distance from BF16 (ΔNLL ~0.002, top-1
94.3 %). The one honest caveat on the whole package remains the phase-1 §10.4 finding about F,
which is the incumbent's problem, not this PR's.
