# PCRL hard-5 balanced gain screen

All gains use the same C/seed-12 checkpoint, profiles, scales and evaluation
seeds `79500..79509` (320 rows per gain).

| gain | overall L1 | primary-five L1 | held-out-three L1 | invalid actions |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.230639 | 0.259182 | 0.183067 | 0.209375 |
| 2 | 0.219864 | 0.251349 | 0.167388 | 0.225000 |
| 3 | 0.208367 | 0.241336 | 0.153420 | 0.246875 |
| 4 | 0.200626 | 0.231046 | 0.149925 | 0.262500 |

Gain 4 improves primary-five L1 by only 10.86% versus gain 1. Search is
strictly positive in only 10/40 paired cells and is mainly driven by the 4x24
scale. Deadline, makespan and coverage remain almost unchanged, but invalid
actions rise by 25.4%.

Conclusion: inference-time gain scaling is useful diagnostic evidence but does
not establish the 30% controllability gate. It cannot unlock pilot, formal,
JEPA or world-model work.

