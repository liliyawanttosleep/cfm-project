# C2 summary (seed 0, FID on 10k samples, noise seed 100)

## OT-CFM  (dopri5 FID 11.5 at ~271 NFE; κ peak t = 1.00, share of ∫κ in t > 0.9 = 0.69, G = 0.50)

| solver | NFE | uniform | √κ | calibrated (endpoint) | best fixed grid | endpoint err: uniform → √κ → calibrated |
|---|---|---|---|---|---|---|
| euler | 2 | 76.0 | 86.3 | 108.8 (shift_0.2973) | 76.0 (uniform) | 0.286 → 0.305 → 0.226 |
| euler | 4 | 32.3 | 32.8 | 50.5 (shift_0.2973) | 32.3 (uniform) | 0.210 → 0.205 → 0.150 |
| euler | 8 | 20.5 | 19.7 | 29.5 (shift_0.25) | 19.7 (sqrt_kappa) | 0.140 → 0.138 → 0.101 |
| euler | 16 | 17.4 | 14.3 | 19.2 (shift_0.25) | 14.3 (sqrt_kappa) | 0.098 → 0.095 → 0.062 |
| euler | 32 | 16.2 | 12.5 | 16.3 (shift_0.2102) | 12.5 (sqrt_kappa) | 0.066 → 0.061 → 0.038 |
| heun | 2 | 349.6 | – | 349.6 (uniform) | 349.6 (uniform) | 0.685 → – → 0.685 |
| heun | 4 | 226.4 | 214.8 | 171.4 (shift_3.364) | 182.0 (shift_4) | 0.429 → 0.418 → 0.366 |
| heun | 8 | 134.4 | 61.1 | 61.1 (sqrt_kappa) | 61.1 (sqrt_kappa) | 0.225 → 0.173 → 0.173 |
| heun | 16 | 83.3 | 22.8 | 22.8 (sqrt_kappa) | 22.8 (sqrt_kappa) | 0.111 → 0.081 → 0.081 |
| heun | 32 | 50.4 | 13.3 | 13.3 (sqrt_kappa) | 13.3 (sqrt_kappa) | 0.046 → 0.032 → 0.032 |
| rk4 | 4 | 234.2 | – | 234.2 (uniform) | 234.2 (uniform) | 0.339 → – → 0.339 |
| rk4 | 8 | 113.7 | 97.0 | 149.7 (shift_0.5) | 52.1 (shift_4) | 0.185 → 0.207 → 0.177 |
| rk4 | 16 | 66.1 | 31.8 | 31.8 (sqrt_kappa) | 30.0 (shift_4) | 0.077 → 0.062 → 0.062 |
| rk4 | 32 | 40.7 | 12.8 | 12.8 (sqrt_kappa) | 12.8 (sqrt_kappa) | 0.026 → 0.018 → 0.018 |

## I-CFM  (dopri5 FID 12.1 at ~280 NFE; κ peak t = 1.00, share of ∫κ in t > 0.9 = 0.65, G = 0.54)

| solver | NFE | uniform | √κ | calibrated (endpoint) | best fixed grid | endpoint err: uniform → √κ → calibrated |
|---|---|---|---|---|---|---|
| euler | 2 | 131.9 | 132.7 | 165.5 (shift_0.4204) | 131.9 (uniform) | 0.367 → 0.383 → 0.307 |
| euler | 4 | 56.6 | 60.6 | 69.6 (shift_0.3536) | 55.6 (shift_0.5) | 0.246 → 0.251 → 0.197 |
| euler | 8 | 32.0 | 31.3 | 35.8 (shift_0.3536) | 31.3 (sqrt_kappa) | 0.166 → 0.167 → 0.131 |
| euler | 16 | 22.4 | 19.8 | 23.5 (shift_0.3536) | 19.8 (sqrt_kappa) | 0.109 → 0.109 → 0.082 |
| euler | 32 | 18.4 | 15.2 | 20.3 (shift_0.2102) | 15.2 (sqrt_kappa) | 0.068 → 0.070 → 0.056 |
| heun | 2 | 362.9 | – | 362.9 (uniform) | 362.9 (uniform) | 0.723 → – → 0.723 |
| heun | 4 | 213.4 | 207.6 | 256.5 (shift_0.5) | 187.0 (shift_2) | 0.462 → 0.472 → 0.416 |
| heun | 8 | 126.4 | 61.9 | 61.9 (sqrt_kappa) | 61.9 (sqrt_kappa) | 0.211 → 0.178 → 0.178 |
| heun | 16 | 78.4 | 22.1 | 22.1 (sqrt_kappa) | 22.1 (sqrt_kappa) | 0.094 → 0.070 → 0.070 |
| heun | 32 | 48.0 | 13.7 | 13.7 (sqrt_kappa) | 13.7 (sqrt_kappa) | 0.045 → 0.027 → 0.027 |
| rk4 | 4 | 236.7 | – | 236.7 (uniform) | 236.7 (uniform) | 0.400 → – → 0.400 |
| rk4 | 8 | 124.7 | 113.9 | 146.9 (shift_0.7071) | 56.5 (shift_4) | 0.158 → 0.179 → 0.151 |
| rk4 | 16 | 65.9 | 31.3 | 31.3 (sqrt_kappa) | 28.8 (shift_4) | 0.065 → 0.055 → 0.055 |
| rk4 | 32 | 37.1 | 13.2 | 13.2 (sqrt_kappa) | 13.2 (sqrt_kappa) | 0.024 → 0.014 → 0.014 |

## Steps to reach a target FID (interpolated in log NFE)

| model | sampler | NFE@20 | NFE@1.1×dopri5 |
|---|---|---|---|
| OT-CFM | Euler, uniform (E2 grid 1–100) | 9.2 | – |
| OT-CFM | euler, uniform (C2 grid 2–32) | 8.9 | – |
| OT-CFM | euler, sqrt_kappa (C2 grid 2–32) | 7.9 | 30.3 |
| OT-CFM | heun, uniform (C2 grid 2–32) | – | – |
| OT-CFM | heun, sqrt_kappa (C2 grid 2–32) | 19.6 | – |
| OT-CFM | rk4, uniform (C2 grid 2–32) | – | – |
| OT-CFM | rk4, sqrt_kappa (C2 grid 2–32) | 24.6 | – |
| OT-CFM | heun_e (last step avoids t = 1), uniform | 6.6 | – |
| OT-CFM | heun_e (last step avoids t = 1), sqrt_kappa | 6.7 | 29.7 |
| OT-CFM | rk4_e (last step avoids t = 1), uniform | 7.7 | – |
| I-CFM | Euler, uniform (E2 grid 1–100) | 23.7 | – |
| I-CFM | euler, uniform (C2 grid 2–32) | 24.3 | – |
| I-CFM | euler, sqrt_kappa (C2 grid 2–32) | 15.8 | – |
| I-CFM | heun, uniform (C2 grid 2–32) | – | – |
| I-CFM | heun, sqrt_kappa (C2 grid 2–32) | 19.0 | – |
| I-CFM | rk4, uniform (C2 grid 2–32) | – | – |
| I-CFM | rk4, sqrt_kappa (C2 grid 2–32) | 24.6 | 31.8 |
| I-CFM | heun_e (last step avoids t = 1), uniform | 11.4 | – |
| I-CFM | heun_e (last step avoids t = 1), sqrt_kappa | 10.2 | 28.2 |
| I-CFM | rk4_e (last step avoids t = 1), uniform | 9.9 | – |

## t = 1 check, OT-CFM

```
{
 "rms_x": 0.9620916247367859,
 "real_rms_v1_minus_x": 0.4095116411769107,
 "real_rms_v0.9_minus_target": 0.3557218976938079,
 "real_rms_v0.98_minus_target": 0.3847219872889869,
 "real_rms_v0.99_minus_target": 0.40140720948638514,
 "real_rms_v0.999_minus_target": 0.5691753694458614,
 "gen_rms_v1_minus_x": 0.8050991892814636,
 "gen_dv_dt_0.99_to_1": 0.5702894926071167,
 "gen_dv_dt_0.9_to_0.91": 1.116301417350769
}
```

| setting (solver|NFE|grid) | endpoint err | FID |
|---|---|---|
| heun_e|3|uniform | 0.279 | 41.6 |
| heun_e|3|sqrt_kappa | 0.307 | 37.2 |
| heun_e|7|uniform | 0.175 | 18.4 |
| heun_e|7|sqrt_kappa | 0.167 | 19.0 |
| heun_e|15|uniform | 0.089 | 15.5 |
| heun_e|15|sqrt_kappa | 0.081 | 15.6 |
| heun_e|31|uniform | 0.034 | 15.9 |
| heun_e|31|sqrt_kappa | 0.032 | 12.4 |
| rk4_e|6|uniform | 0.163 | 21.5 |
| rk4_e|14|uniform | 0.065 | 16.4 |
| rk4_e|30|uniform | 0.018 | 16.5 |

## t = 1 check, I-CFM

```
{
 "rms_x": 0.9620916247367859,
 "real_rms_v1_minus_x": 0.40505775952855033,
 "real_rms_v0.9_minus_target": 0.3513774722140168,
 "real_rms_v0.98_minus_target": 0.3793402206587207,
 "real_rms_v0.99_minus_target": 0.39452970928169556,
 "real_rms_v0.999_minus_target": 0.5691722224369753,
 "gen_rms_v1_minus_x": 0.8355749845504761,
 "gen_dv_dt_0.99_to_1": 0.4512104094028473,
 "gen_dv_dt_0.9_to_0.91": 0.9133433103561401
}
```

| setting (solver|NFE|grid) | endpoint err | FID |
|---|---|---|
| heun_e|3|uniform | 0.359 | 66.5 |
| heun_e|3|sqrt_kappa | 0.382 | 59.6 |
| heun_e|7|uniform | 0.162 | 23.9 |
| heun_e|7|sqrt_kappa | 0.172 | 23.8 |
| heun_e|15|uniform | 0.066 | 17.8 |
| heun_e|15|sqrt_kappa | 0.070 | 16.0 |
| heun_e|31|uniform | 0.031 | 16.5 |
| heun_e|31|sqrt_kappa | 0.027 | 12.9 |
| rk4_e|6|uniform | 0.135 | 23.6 |
| rk4_e|14|uniform | 0.049 | 17.5 |
| rk4_e|30|uniform | 0.014 | 16.3 |
