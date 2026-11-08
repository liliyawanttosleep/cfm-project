# Derivations for the final report

> Written 2026-11-07. Source for **Section 3 (Methods)** and **Appendix A** of the report; see `report/README.md` for where each part goes.
> Each derivation states what it is used for and which experiment in `docs/FINAL_REPORT_PLAN.md` tests it.
> D1–D4 restate what the talk already showed (slides 9–16, 27, backup B1–B2). D5–D8 are new and support the final experiments.

## D0. Notation

- Noise $x_0 \sim p_0 = \mathcal N(0, I_d)$, data $x_1 \sim p_1$; a coupling $q(x_0, x_1)$ with these marginals. Independent coupling: $q = p_0 \otimes p_1$ (I-CFM). Minibatch OT coupling: OT-CFM.
- Straight conditional path $x_t = (1-t)\,x_0 + t\,x_1$, conditional velocity $Y = \dot x_t = x_1 - x_0$. Time runs from noise ($t=0$) to data ($t=1$).
- Marginal density $p_t$ = law of $x_t$; marginal velocity $u_t(x) = \mathbb E[\,x_1 - x_0 \mid x_t = x\,]$.
- Network $v_\theta(x, t)$; sampling solves $\dot x = v_\theta(x, t)$ from $x(0) \sim p_0$. Euler with grid $0 = t_0 < \dots < t_N = 1$, steps $h_i = t_{i+1} - t_i$; cost = number of network evaluations (NFE).

---

## D1. The conditional loss has the same gradient as the marginal loss

**Used for**: why regressing on a random target works (Methods 3.1; Appendix A.1).

$$
\mathcal L_{FM}(\theta) = \mathbb E_{t,\,x \sim p_t}\,\|v_\theta(x,t) - u_t(x)\|^2,
\qquad
\mathcal L_{CFM}(\theta) = \mathbb E_{t,\,(x_0,x_1) \sim q}\,\|v_\theta(x_t,t) - (x_1 - x_0)\|^2 .
$$

Fix $t$ and write $V = v_\theta(x_t, t)$, which depends on $(x_0, x_1)$ only through $x_t$. Then

$$
\|V - Y\|^2 = \|V\|^2 - 2\langle V, Y\rangle + \|Y\|^2,
\qquad
\mathbb E\langle V, Y\rangle = \mathbb E\langle V, \mathbb E[Y \mid x_t]\rangle = \mathbb E\langle V, u_t(x_t)\rangle
$$

by the tower rule. Doing the same expansion for $\mathcal L_{FM}$ and subtracting,

$$
\mathcal L_{CFM}(\theta) = \mathcal L_{FM}(\theta) + C,
\qquad
C = \mathbb E_t\,\mathbb E\big[\operatorname{tr}\operatorname{Cov}(x_1 - x_0 \mid x_t)\big] \ge 0,
$$

so $\nabla_\theta \mathcal L_{CFM} = \nabla_\theta \mathcal L_{FM}$, and for every $t$ the minimiser is the posterior mean $v^\star(\cdot, t) = u_t$.

**Consequences**
- $C$ does not depend on $\theta$, so the CFM loss cannot go to 0. $C$ depends on the coupling, so **losses of I-CFM and OT-CFM are not comparable** (talk: final loss 0.168 vs 0.118 on MNIST).
- Assumption: $\mathbb E\|x_1 - x_0\|^2 < \infty$. Lipman et al. (2023), Theorem 2, state the same result (they also assume $p_t > 0$).

## D2. The marginal velocity transports $p_0$ to $p_1$

**Used for**: why integrating the learned field gives data (Methods 3.1; Appendix A.2).

For any smooth, compactly supported test function $\varphi$,

$$
\frac{d}{dt}\,\mathbb E\,\varphi(x_t)
= \mathbb E\,\langle \nabla\varphi(x_t),\, x_1 - x_0\rangle
= \mathbb E\,\langle \nabla\varphi(x_t),\, u_t(x_t)\rangle ,
$$

again by the tower rule. This is the weak form of the continuity equation $\partial_t p_t + \nabla\!\cdot(p_t\, u_t) = 0$. If $u_t$ is regular enough for the ODE $\dot x = u_t(x)$ to have unique solutions, the law of the ODE solution started from $p_0$ solves the same equation with the same initial condition, hence equals $p_t$ for all $t$, and $p_1$ at $t = 1$ (Lipman et al., 2023, Theorem 1; Liu et al., 2023, Theorem 3.3).

## D3. Where one step lands: the posterior mean

**Used for**: why I-CFM's one-step samples are blurry (Methods 3.2; talk slides 16 and 25).

Since $x_0 = (x_t - t\,x_1)/(1-t)$, the target can be rewritten as $x_1 - x_0 = (x_1 - x_t)/(1-t)$, so

$$
u_t(x) = \frac{\hat x_1(x, t) - x}{1 - t},
\qquad \hat x_1(x, t) = \mathbb E[x_1 \mid x_t = x].
$$

**One Euler step from $t$ straight to $t = 1$** ($h = 1 - t$) gives $x + (1-t)\,u_t(x) = \hat x_1(x, t)$: it always lands on the posterior mean of the data given the current point.

At $t = 0$ we have $x_t = x_0$, so a one-step sample is $\mathbb E[x_1 \mid x_0]$:
- **Independent coupling**: $x_1 \perp x_0$, so $\mathbb E[x_1 \mid x_0] = \mathbb E[x_1]$. *Every noise vector is mapped to the data mean.* On MNIST, I-CFM one-step samples are 0.04 (RMS per pixel) from the data mean, versus 0.52 for a real digit.
- **OT coupling**: $\mathbb E[x_1 \mid x_0]$ depends on $x_0$, so one step already separates samples (OT-CFM: 0.22 from the mean). It is still an average over the data points paired with nearby noise, hence blurry when the coupling is far from deterministic.

So few-step quality depends on how close the coupling is to a **deterministic** map $x_0 \mapsto x_1$. This motivates OT (D4) and reflow (D7).

## D4. How much any pairing can save in high dimension

**Used for**: why minibatch OT barely changes the pairs at $d = 784$ (Methods 3.3; experiment C3; talk slide 27).

For any coupling with $\mathbb E\,x_0 = 0$,

$$
\mathbb E\|x_0 - x_1\|^2 = \mathbb E\|x_0\|^2 + \mathbb E\|x_1\|^2 - 2\,\mathbb E\langle x_0, x_1\rangle ,
$$

and the independent coupling has $\mathbb E\langle x_0, x_1\rangle = 0$. So the **saving** of a coupling relative to independent pairs is $2\,\mathbb E\langle x_0, x_1\rangle$. Let $\mu, \Sigma$ be the mean and covariance of $p_1$. The Gelbrich bound (Gelbrich, 1990) gives, for any coupling,

$$
\mathbb E\|x_0 - x_1\|^2 \;\ge\; \|\mu\|^2 + \operatorname{tr}\big(I + \Sigma - 2\,\Sigma^{1/2}\big),
$$

while the independent cost is $d + \|\mu\|^2 + \operatorname{tr}\Sigma$. Subtracting,

$$
\text{saving} \;\le\; 2\,\operatorname{tr}\Sigma^{1/2},
$$

with equality when $p_1$ is Gaussian and the coupling is the OT map.

**On MNIST** (pixels in $[-1, 1]$): independent cost $\approx 1{,}509$, bound $2\,\operatorname{tr}\Sigma^{1/2} = 386$ (at most a 26% saving), measured minibatch-OT saving at batch 128 = 66 (4%). The bound is far below the 85–94% savings in 2D, because most of $\mathbb E\|x_0\|^2 = d$ is isotropic noise that no pairing can cancel. C3 measures how the saving grows with the OT batch size.

## D5. A time grid from the curvature of the paths (our method)

**Used for**: the curvature-adapted grid (Methods 3.3; experiment C2).

**Local error.** Let $x(t)$ be an exact trajectory of $\dot x = v(x, t)$ and define its acceleration

$$
a(t) = \frac{d}{dt}\,v(x(t), t) = \partial_t v + (\nabla_x v)\,v .
$$

One Euler step of size $h$ from $t$ has local error

$$
x(t+h) - x(t) - h\,v(x(t), t) = \int_t^{t+h}\big(v(x(s), s) - v(x(t), t)\big)\,ds \approx \tfrac{h^2}{2}\,a(t).
$$

Let $\kappa(t) = \mathbb E\,\|a(t)\|$, averaged over trajectories ($x(0) \sim p_0$). $\kappa \equiv 0$ means every path is a straight line at constant speed (compare D7).

**Optimal step density.** Ignore how errors are amplified along the trajectory, and add up the local errors. With a step density $n(t)$ (steps per unit time, so $h(t) = 1/n(t)$),

$$
E(n) = \sum_i \tfrac{h_i^2}{2}\,\kappa(t_i) \approx \int_0^1 \frac{\kappa(t)}{2\,n(t)}\,dt,
\qquad \text{subject to } \int_0^1 n(t)\,dt = N .
$$

The Lagrangian condition $-\kappa/(2n^2) + \lambda = 0$ gives

$$
n^\star(t) = N\,\frac{\sqrt{\kappa(t)}}{\int_0^1 \sqrt{\kappa}},
\qquad
t_i = F^{-1}(i/N),\quad F(t) = \frac{\int_0^t \sqrt{\kappa}}{\int_0^1 \sqrt{\kappa}} .
$$

**Each step covers the same amount of $\sqrt\kappa$**: steps are dense where paths bend.

**Predicted gain.** The minimal and the uniform-grid errors are

$$
E(n^\star) = \frac{\big(\int_0^1 \sqrt\kappa\big)^2}{2N},
\qquad
E(\text{uniform}) = \frac{\int_0^1 \kappa}{2N},
\qquad
G = \frac{E(n^\star)}{E(\text{uniform})} = \frac{\big(\int_0^1 \sqrt\kappa\big)^2}{\int_0^1 \kappa} \le 1
$$

by Cauchy–Schwarz, with equality only if $\kappa$ is constant. **$G$ is computed from the measured $\kappa$ before running the comparison**, so C2 can test the prediction: the ratio of endpoint errors (adapted vs uniform), measured against a dopri5 reference from the same $x_0$, should be close to $G$ at moderate $N$.

**Practical notes**
- Estimate $\kappa$ once per model from 256 dopri5 trajectories, by finite differences of $v(x(t), t)$ on a fine time grid. Smooth it and add a small floor $\epsilon$ so that no step becomes infinitely long.
- The derivation is a first-order heuristic. It ignores the amplification of earlier errors (a factor up to $e^{\int_t^1 L}$ for a field with Lipschitz constant $L$), and the error of the network itself.
- For a method of order $p$ (local error $\propto h^{p+1}\|x^{(p+1)}\|$) the same argument gives $n^\star \propto \|x^{(p+1)}\|^{1/(p+1)}$. For Heun ($p = 2$) we reuse the Euler grid as a heuristic and say so.
- Baselines in C2: the uniform grid, and the shift grid $t' = s\,t/(1 + (s-1)\,t)$ (the timestep shift of Esser et al., 2024, written in our time direction). $s > 1$ moves grid points towards $t = 1$, so steps are denser near the data end; $s < 1$ makes them denser near the noise end.

**Numerical check on 1D examples** (`x₀ ~ N(0, 1)`, exact field from D3, Euler endpoint RMS error against a fine RK4 solution, 4,000 paths; `python code/c2_solvers/check_derivations.py`):

| Target | Predicted $G$ | Grid | N = 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| $\tfrac12\mathcal N(\pm 2, 0.5^2)$ (smooth) | 0.95 | uniform | **0.762** | 0.354 | 0.179 | 0.091 |
| | | $\sqrt\kappa$ | 0.798 | **0.321** | **0.164** | **0.083** |
| | | best shift ($s = 0.5$) | 0.911 | 0.331 | **0.164** | 0.084 |
| $\pm 2$ with prob. ½ (two points) | 0.63 | uniform | 0.770 | 0.106 | **0.000** | **0.000** |
| | | $\sqrt\kappa$ | 1.128 | 0.644 | 0.360 | 0.188 |
| | | best shift ($s = 4$) | **0.390** | **0.000** | **0.000** | **0.000** |

- **Smooth target**: the $\sqrt\kappa$ grid lowers the error by 8–9% from N = 4 on; the prediction $1 - G = 5\%$ has the right size. At N = 2 it is slightly worse.
- **Two-point target**: the $\sqrt\kappa$ grid is much *worse*. $\kappa$ peaks at $t \approx 0.33$ and is about 1000× smaller near $t = 1$, so the grid takes one long last step. By D3 that step lands on $\mathbb E[x_1 \mid x_t]$, which is still an average when the posterior has not yet sharpened. This error is not a small-step Taylor error, so the derivation does not see it. The paths are straight near $t = 1$, but the field changes sharply *off* the path (large Jacobian), which is the amplification term we dropped.
- **No single fixed rule wins**: the best shift is $s = 0.5$ (noise end) for the smooth target and $s = 4$ (data end) for the two-point target.

**Consequence for C2.** The $\sqrt\kappa$ grid is a hypothesis to test, not the method we rely on. MNIST lies between the two cases: its data are concentrated, but the model's distribution is continuous. The practical method is a **calibrated grid**:
1. For each NFE, generate 256 calibration paths with dopri5.
2. Run each candidate grid (uniform, shift $s \in \{0.25, 0.5, 2, 4\}$, $\sqrt\kappa$) from the same $x_0$ and measure the endpoint error.
3. Use the grid with the smallest error for the FID run.

This costs a few seconds per model and uses no real data. C2 reports the endpoint error and FID of every candidate, so the report can say which rule wins on MNIST and whether endpoint error predicts FID.

## D6. Changing how $t$ is sampled in training is a weighted loss

**Used for**: the training-time distribution (Methods 3.3; experiment C4).

Sampling $t \sim \pi$ instead of $U[0, 1]$ minimises

$$
\mathcal L_\pi(\theta) = \int_0^1 \pi(t)\;\mathbb E\,\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\,dt
= \int_0^1 \pi(t)\,\big(\epsilon_\theta(t) + C_t\big)\,dt,
$$

where $\epsilon_\theta(t) = \mathbb E_{x \sim p_t}\|v_\theta(x, t) - u_t(x)\|^2$ and $C_t$ is the per-$t$ floor from D1.

- **With unlimited capacity, $\pi$ changes nothing**: for every $t$ with $\pi(t) > 0$ the minimiser is still $u_t$. So $\pi$ must be positive on all of $[0, 1]$.
- **With a finite network and finite training, $\pi$ decides where $\epsilon_\theta(t)$ is small.** Why this matters for sampling: by Grönwall's inequality, for an exact solution of the learned ODE,

$$
\|x_\theta(1) - x(1)\| \le \int_0^1 e^{\int_t^1 L(s)\,ds}\;\|v_\theta(x(t), t) - u_t(x(t))\|\,dt ,
$$

  so errors at different times are amplified differently, and the discretisation error of D5 adds to this.
- **Variants tested in C4**:
  - logit-normal $\pi(t) = \frac{1}{s\sqrt{2\pi}}\,\frac{1}{t(1-t)}\exp\!\big(-\frac{(\operatorname{logit} t - m)^2}{2s^2}\big)$ with $(m, s) = (0, 1)$, the default recommended by Esser et al. (2024). They also report that $(0.5, 0.6)$ is best at 50 sampling steps but much worse at 5 steps, so a few-step gain is not guaranteed.
  - Curvature-weighted mixture $\pi(t) = (1-\alpha) + \alpha\,\kappa(t) / \int_0^1 \kappa$ with $\alpha = 0.5$, which keeps $\pi > 0$ and trains more where the paths bend (D5).

## D7. Straightness, one-step generation and reflow

**Used for**: reflow (Methods 3.3; experiment C5) and the straightness diagnostic (C6).

For a flow $\dot Z_t = v(Z_t, t)$, Liu et al. (2023) define the straightness

$$
S(Z) = \int_0^1 \mathbb E\,\big\|(Z_1 - Z_0) - \dot Z_t\big\|^2\,dt .
$$

**$S = 0$ makes one Euler step exact.** If $S = 0$, then $\dot Z_t = Z_1 - Z_0$ for almost every $t$, so $Z_t = Z_0 + t\,(Z_1 - Z_0)$ is a straight line at constant speed, and by continuity $v(Z_0, 0) = Z_1 - Z_0$. Hence $Z_0 + v(Z_0, 0) = Z_1$. Equivalently, the acceleration of D5 vanishes: $\kappa \equiv 0$.

**Our path-length ratio** $R = \mathbb E\big[\int_0^1 \|\dot Z_t\|\,dt \,/\, \|Z_1 - Z_0\|\big] \ge 1$ equals 1 when paths are straight in **direction**. It does not detect changes of speed, so it is a weaker diagnostic than $S$. C1 and C6 report both.

**Reflow.** Draw $x_0 \sim p_0$, solve the learned ODE to get $\hat x_1$, and retrain CFM on the deterministic pairs $(x_0, \hat x_1)$. For an exactly learned field, Liu et al. (2023) prove:
- (Theorem 3.3) the new flow has the same time marginals, so it still maps $p_0$ to $p_1$;
- (Theorem 3.5) $\mathbb E\,c(Z_1 - Z_0) \le \mathbb E\,c(X_1 - X_0)$ for every convex cost $c$;
- (Theorem 3.7) after $K$ reflows, $\min_{k \le K} S(Z^k) \le \mathbb E\|X_1 - X_0\|^2 / K$.

Connection to D3: the reflow coupling is deterministic, so $\mathbb E[x_1 \mid x_0] = \hat x_1(x_0)$ and a one-step sample is no longer an average.

**Caveat**: with a learned field, $\hat x_1$ follows the *model's* distribution, not $p_1$. Its errors are inherited by the reflowed model, which is why C5 checks many-step FID and class coverage as well as one-step FID. Training-cost accounting for reflow = time to generate the pairs + time to retrain.

## D8. Statistics of the evaluation

**Used for**: the evaluation protocol (Methods 3.4) and the reliability experiment (C1).

**FID.** With Inception features of real and generated images summarised by $(\mu_r, \Sigma_r)$ and $(\mu_g, \Sigma_g)$,

$$
\mathrm{FID} = \|\mu_r - \mu_g\|^2 + \operatorname{tr}\big(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2}\big)
= W_2^2\big(\mathcal N(\mu_r, \Sigma_r),\, \mathcal N(\mu_g, \Sigma_g)\big),
$$

the same Gaussian $W_2$ formula as in D4, applied in feature space. FID estimated from $n$ samples is biased upward. Chong & Forsyth (2020) show the bias is approximately linear in $1/n$, so C1 fits FID against $1/n$ (2k–20k samples) to estimate it.

**Two sources of variation.** For training seed $s$ and sampling-noise draw $r$, model the score as $F_{s,r} = \mu + a_s + e_{s,r}$, with $\operatorname{Var} a_s = \sigma^2_{\text{seed}}$ and $\operatorname{Var} e_{s,r} = \sigma^2_{\text{noise}}$.
- $\sigma^2_{\text{noise}}$: from repeated noise draws of one trained model.
- $\sigma^2_{\text{seed}} \approx s^2_{\text{between}} - \sigma^2_{\text{noise}}$, where $s^2_{\text{between}}$ is the sample variance of single-draw scores across seeds.
- With 3 seeds this has 2 degrees of freedom and is only a rough guide.

**Comparing methods.**
- Methods are compared **paired**: same seed index and the same sampling noise. Common random numbers remove most of the noise term from the difference $D_s = F^A_s - F^B_s$.
- With 3 seeds the smallest attainable one-sided sign-test p-value is $0.5^3 = 0.125$, so we report the three $D_s$ and whether they share a sign, never "significant".
- **Improvement threshold**: a difference counts only if it exceeds $2\times$ the standard deviation of single-run scores across training seeds.

**Class coverage (the course method).**
- k-means with $k$ centres is fitted on $m$ real images only; each generated image is assigned to its nearest centre, giving shares $\hat p_k$ from $n$ samples. The reference shares $p_k$ come from the $m$ real images.
- We report

$$
\mathrm{TV} = \tfrac12 \sum_k |\hat p_k - p_k|,
\qquad
\chi^2 = \sum_k \frac{(O_k - E_k)^2}{E_k},
\qquad
w = \sqrt{\chi^2 / n},
$$

  with $O_k = n\hat p_k$ and $E_k = n p_k$. $w$ is Cohen's effect size.
- **Noise floor of TV.** Even if the generated distribution were exactly right, $\hat p_k - p_k$ is approximately $\mathcal N\big(0,\, p_k(1-p_k)(1/n + 1/m)\big)$, and $\mathbb E|Z| = \sigma\sqrt{2/\pi}$ for $Z \sim \mathcal N(0, \sigma^2)$, so

$$
\mathbb E[\mathrm{TV}] \approx \tfrac12 \sum_k \sqrt{\tfrac{2}{\pi}\,p_k(1-p_k)\big(\tfrac1n + \tfrac1m\big)} .
$$

  For $k = 10$, $p_k \approx 0.1$, $n = 10{,}000$ and $m = 30{,}000$, this is about 0.014. The real-test-set baseline is TV = 0.017, so the baseline is mostly sampling noise. Generated models at 2.6–11× the baseline are well above it.
- Because $p_k$ is itself estimated, the correct test is a two-sample χ² test of homogeneity (generated vs real counts), not a goodness-of-fit test against fixed $p_k$. C1 switches to it.
- **Bootstrap interval**: resample generated and reference assignments with replacement ($B = 1{,}000$), recompute TV, and take the 2.5% and 97.5% percentiles.

---

## References for these derivations

- Lipman, Chen, Ben-Hamu, Nickel, Le. *Flow Matching for Generative Modeling.* ICLR 2023. (Theorems 1–2)
- Liu, Gong, Liu. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.* ICLR 2023. (Straightness; Theorems 3.3, 3.5, 3.7)
- Tong et al. *Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport.* TMLR 2024.
- Esser et al. *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis.* ICML 2024. (Logit-normal and shifted timesteps)
- Gelbrich. *On a formula for the L2 Wasserstein metric between measures on Euclidean and Hilbert spaces.* Math. Nachr. 1990.
- Chong, Forsyth. *Effectively Unbiased FID and Inception Score and Where to Find Them.* CVPR 2020.
- Heusel et al. *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.* NeurIPS 2017. (FID)
