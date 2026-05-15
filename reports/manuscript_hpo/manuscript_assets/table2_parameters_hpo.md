# Table 2 — Neural parameter counts (seed 11, HPO sweep)

| Component | Architecture | Neural parameters |
|---|---|---:|
| Grey-box closure MLP | $3 \rightarrow 128 \rightarrow 64 \rightarrow 32 \rightarrow 1$ | 11,250 |
| Shared initial-state network | $6 \rightarrow 32 \rightarrow 4$ | 356 |
| Matched-input black-box MLP | $4 \rightarrow 32 \rightarrow 16 \rightarrow 1$ | 705 |
| Residual LSTM correction | LSTM(7, 32) plus $32 \rightarrow 32 \rightarrow 1$ readout | 6,337 |
