# v6 / v7 / v11 / v12 비교

| 버전 | 파일 | Q·Kᵀ 누산 | PV 누산 | spill (vision / prefill·denoise) | vision | prefill | denoise | 평평 · prefill | 뾰족 · vision | 뾰족 · prefill | 뾰족 · denoise |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OpenVINO | `sdpa_opt.cl` | fp16 | fp16 (정규화한 P) | 확인 안 됨 | 1.60 | 5.04 | 0.76 | 4.1e-4 / 5.4e-3 | 6.3e-4 / 1.7e-2 | 1.1e-3 / 2.5e-2 | 1.2e-3 / 1.9e-2 |
| v6 | [attention-fa-intel6.comp](attention-fa-intel6.comp) | fp16 | fp16 | 0 / 0 B | **0.491 ms** (−69%) | **4.23 ms** (−16%) | **0.371 ms** (−51%) | 6.1e-4 / 1.5e-2 | 6.5e-4 / 1.7e-2 | 1.1e-3 / 2.5e-2 | 1.2e-3 / 1.9e-2 |
| v7 | [attention-fa-intel7.comp](attention-fa-intel7.comp) | fp16 | fp32 | 320 / 1184 B | 0.632 (−60%) | 5.27 (+5%) | 0.504 (−34%) | 1.6e-4 / 1.3e-3 | 6.0e-4 / 1.7e-2 | 1.1e-3 / 2.5e-2 | 1.2e-3 / 1.9e-2 |
| v11 | [attention-fa-intel11.comp](attention-fa-intel11.comp) | fp32 | fp32 | 2336 / 3040 B | 0.877 (−45%) | 7.52 (+49%) | 0.739 (−3%) | 3.7e-5 / 2.7e-4 | 4.5e-5 / 7.1e-4 | 9.7e-6 / 1.4e-4 | 9.9e-6 / 1.0e-4 |
| v12 | [attention-fa-intel12.comp](attention-fa-intel12.comp) | 청크 안 fp16, 청크끼리 fp32 | fp32 | 896 / 1664 B | 0.675 (−58%) | 5.55 (+10%) | 0.529 (−30%) | 6.3e-5 / 4.7e-4 | 3.0e-4 / 6.4e-3 | 2.1e-4 / 4.2e-3 | 2.2e-4 / 4.3e-3 |

- 공통 조건: 곱셈 fp16, softmax fp32, split-K 없음. 설정은 vision wg256 G1 TQ32 BK128, prefill·denoise wg256 G8 TQ2 BK256. 시간 괄호 안 퍼센트는 OpenVINO 대비다(음수가 더 빠름).
- 오차는 max|O| 대비 중앙값 / 최대값이다.
- 평평은 Q 원본(행당 최대 확률 중앙값 0.003~0.009), 뾰족은 Q × 8(0.18~0.25)이다.
- OpenVINO 시간은 연산별 카운터에서 추정한 다른 시점의 값이라 직접 비교할 수 없다.
