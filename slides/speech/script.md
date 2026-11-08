---
title: "Flow Matching vs. Diffusion: Speech Script"
subtitle: "Wentao Yang and Kaiming Zhang · Class talk, November 10, 2026"
---

# Speaking order

| Part | Slides | Speaker | Time | Script status |
|:-----|:-----|:-----|:-----|:-----|
| Opening | 1–4 | Wentao | 4 min | Written |
| Part I: Generation as transport | 5–8 | Kaiming | 4.5 min | Written |
| Part II: Conditional Flow Matching | 9–14 | Wentao | 9 min | Written |
| Part III: 2D examples | 15–19 | Wentao | @P3@ min | Written |
| Part IV: Link to diffusion | 20–22 | Kaiming | @P4@ min | Written |
| Part V: From 2D to MNIST, and E3 | 23–28 | Kaiming, then Wentao, then Kaiming | @P5@ min | Written |
| Close | 29–31 | Kaiming, then Wentao | @P6@ min | Written |
| Backup | B1–B6 | Either | Not timed | Use table only |

Each slide has two parts: the script to speak, and the formulas to explain. Words in *[brackets]* are cues and are not spoken. Times assume 140 words per minute and have not been rehearsed. The table adds up to about 36.9 minutes. The spoken words alone take about @TOTAL@ minutes, because the Opening's words take @OPEN@ of its 4 minutes. The scripts match the online version of this document.

Throughout the talk, t = 0 is noise and t = 1 is data, x₀ is noise and x₁ is data. The running example is x₀ ~ N(0, 1) and x₁ = ±2.

# Opening: slides 1–4

## Slide 1 · Wentao · 0.3 min · Cover

### Script

Hi everyone, I'm Wentao Yang, and this is Kaiming Zhang. Our project is about flow matching and diffusion, the two families of models behind today's image and video generators.

### Formulas to explain

None on this slide.

## Slide 2 · Wentao · 1.2 min · What we set out to do

### Script

Here is the project in one sentence. We want to explain how flow matching turns generation into a regression problem, and then measure how it trades sample quality against sampling cost compared with diffusion.

*[Point to the left card.]* That gives three goals. First, explain: why does a simple least-squares regression give a correct generative model? This is most of today's talk.

*[Middle card.]* Second, compare: with the same network and the same training budget, how does quality change with the number of network calls per sample?

*[Right card.]* Third, evaluate: is the generated data useful? We use the methods from this course to check whether it can train classifiers and whether it covers every class.

*[Bottom row, left to right.]* We do this at two scales, so we can see what survives as the dimension grows: 2D toy data, then MNIST with 784 dimensions.

### Formulas to explain

None on this slide.

## Slide 3 · Wentao · 1 min · Where the method takes us

### Script

Before any formula, here is where the method takes us. *[Left animation.]* On the left, points start in eight Gaussian clusters and follow a learned velocity field until they form two moons. *[Right animation.]* On the right, the same idea on images: every pixel starts as noise and flows until digits appear. Both are our own runs, and we show the measurements later.

*[Point to the words on the right.]* The important message is that these two examples use the same loss and the same training loop. Only the data and the network change: a small MLP on the left, a U-Net on the right.

### Formulas to explain

None on this slide. Let both animations play once before explaining them.

## Slide 4 · Wentao · 1.5 min · Five lines that train a generative model

### Script

This is essentially the whole training code, adapted from the torchcfm MNIST example. Throughout the talk, x-zero is noise and x-one is data.

*[Walk down the code, one line at a time.]* Take a batch of real data and the same number of noise samples. Pick a random time, take the point on the straight line between each noise sample and its data point, and read off the velocity of that line. Then fit the network to that velocity by least squares. There is no likelihood, no ODE solve during training, and no discriminator.

This looks too simple, and it raises three questions, which organize the rest of the talk.

*[Card 1.]* First, why this target? Given the input, x-one minus x-zero is still random, because many pairs pass through the same point. *[Card 2.]* Second, why does the trained network actually generate data? *[Card 3.]* Third, why does changing how we pair noise with data let sampling take fewer steps?

We answer the first two with theory, and the third with 2D experiments. We then connect the method to diffusion and scale it up to images. Kaiming will start with the basic setup.

### Formulas to explain

No formula yet. Explain the code instead.

| Line | What to say | Connects to |
|:-----------------------|:--------------|:---------|
| `x1 = next(data_iter)` | A batch of real data | x₁ ~ p_data |
| `x0 = torch.randn_like(x1)` | The same number of noise samples | x₀ ~ N(0, I) |
| `t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)` | A random time, the point on the straight line at that time, and the velocity of that line | Derived on slide 9 |
| `loss = torch.mean((model(t, xt) - ut) ** 2)` | Plain least squares | Proved correct on slides 10–13 |
| `loss.backward(); optimizer.step()` | One gradient step | Nothing new |

# Part I: slides 5–8

## Slide 5 · Kaiming · 0.9 min · The goal: push noise forward to data

### Script

Let's set up the problem. We want a map T that pushes simple noise forward to the data distribution. *[Point to T#p₀.]* T-sharp p-zero means the distribution of T of z, when z comes from p-zero. So: draw Gaussian noise, apply T, and the output should look like data. You already know a one-dimensional version: the inverse-CDF transform. In high dimensions there is no such formula, so we learn T.

*[Cards, left to right.]* Normalizing flows build T from invertible layers and train by maximum likelihood, but they need a cheap inverse and log-determinant, which restricts the architecture. Continuous normalizing flows define T through an ODE, which removes that restriction, but likelihood training has to solve the ODE at every step. Flow matching keeps the ODE and replaces likelihood training with regression. Let's see what the ODE gives us.

### Formulas to explain

$$T_{\#} p_0 \approx p_{\text{data}}, \qquad p_0 = N(0, I)$$

| Term | What it means |
|:-----|:-----|
| T | The map we want to learn, from noise to data |
| T#p₀ | The distribution of T(z) when z ~ p₀ |
| p₀ = N(0, I) | Noise we can sample as often as we like |
| ≈ p_data | The outputs should look like real data |

## Slide 6 · Kaiming · 1.3 min · A ResNet with many small steps is an ODE

### Script

You all know residual blocks: the output is the input plus a correction. *[Point to the top formula.]* Here x-k is the state after block k, one over K is a small step size, and u is the network, which also takes the time as input. That is exactly a forward Euler step. As K goes to infinity we get an ODE: dx/dt equals u at x and t. So u is a velocity field: it tells a particle at position x and time t where to move.

*[Figure.]* Here is a simple example that we will use all talk: from a standard Gaussian to two points, at plus and minus two. The arrows are the velocity, and the orange curves are exact solutions. The navy line uses Euler with four steps, so four network calls.

*[Code.]* This gives the sampling algorithm: draw noise, then repeat this one line until time one. The number of network calls is NFE, the number of function evaluations, and it is our cost for the rest of the talk. A straight path can be followed with few steps; a curved path needs more.

### Formulas to explain

$$x_{k+1} = x_k + \frac{1}{K}\, u(x_k, t_k) \quad \underset{K \to \infty}{\longrightarrow} \quad \frac{dx}{dt} = u_t(x), \quad x(0) \sim p_0$$

| Term | What it means | In code |
|:-----|:-----|:-----|
| xₖ | The state after block k | `x` |
| 1/K | A small step size | `/ K` |
| u(xₖ, tₖ) | The correction, which also depends on time | `model(k / K, x)` |
| dx/dt = uₜ(x) | uₜ(x) is the velocity of a particle at position x and time t | The loop as K grows |
| x(0) ~ p₀ | Every particle starts as noise | `x = torch.randn(...)` |

## Slide 7 · Kaiming · 1.2 min · Velocity moves density: the continuity equation

### Script

A velocity field moves individual points, but we care about the whole distribution. The continuity equation connects the two. *[Point to p times u.]* p times u is the flux: how much probability crosses a location per unit time. *[Point to the divergence.]* Its divergence is the net outflow at x. *[Point to the time derivative.]* So the equation says: density at x falls when more mass leaves than enters, and rises when more enters. Mass is only moved, never created or destroyed, as in fluid flow.

*[Figure, t = 0.75.]* Look at t equals 0.75. The arrows point toward the two peaks from both sides, so inflow beats outflow there, and the peaks grow. *[Sweep the panels.]* Over time, one Gaussian becomes two narrow peaks.

Why does this matter for generation? It is our test of correctness. Suppose a path p_t starts at noise and ends at data, and a velocity u_t satisfies this equation with it. Then integrating u_t from noise gives data samples. We never compute this equation in code. We use it once, later, to prove that our regression target is right.

### Formulas to explain

$$\partial_t p_t(x) + \nabla \cdot \big( p_t(x)\, u_t(x) \big) = 0$$

| Term | What it means | Say it as |
|:-----|:-----|:-----|
| ∂ₜ pₜ(x) | How fast the density at x changes | "Is density at x going up or down?" |
| pₜ(x) uₜ(x) | The flux: density times velocity | "How much mass crosses x per unit time" |
| ∇ · ( ) | The net outflow at x | "Mass leaving minus mass entering" |
| = 0 | Conservation of mass | "Density falls only if more leaves than enters" |

## Slide 8 · Kaiming · 1.1 min · The ideal loss, and why we cannot compute it

### Script

If we knew the right velocity field, training would be plain regression. *[Point to the formula.]* Read this as: pick a random time, pick a point from the path at that time, and penalize the squared gap between the network's velocity and the true velocity. If this loss reaches zero, the network equals the true field along the path, and by the continuity equation, integrating it from noise gives data.

But we cannot write this loss in code, for two reasons. *[Unknown card.]* First, the target is unknown: we only have samples from the data distribution, so we know neither p_t nor u_t. *[Figure, then Not unique card.]* Second, the target is not defined until we choose a path. These three paths have the same two endpoints, and each would give a different velocity field.

So we need a path that we choose ourselves, with a target we can compute. Wentao will now build one, starting from a single pair of points.

### Formulas to explain

$$L_{FM}(\theta) = \mathbb{E}_{t,\; x \sim p_t} \big\| v_\theta(x, t) - u_t(x) \big\|^2$$

| Term | What it means |
|:-----|:-----|
| E over t, x ~ pₜ | A random time, and a point from the path at that time |
| vθ(x, t) | The network's velocity |
| uₜ(x) | The true velocity. This is the problem: we do not have it |
| ‖ · ‖² | Plain least squares |

# Part II: slides 9–14

## Slide 9 · Wentao · 1.2 min · Step 1: one pair, a straight line

### Script

Thanks, Kaiming. We now build the target in six steps, shown in the bar at the top. We keep the same small example throughout: Gaussian noise, and data at plus and minus two.

*[Point to the top formula.]* Step one is the easiest possible case. Forget the distribution and take a single pair: one noise point x-zero and one data point x-one. *[Second formula.]* Join them with a straight line at constant speed. At t equals 0 we are at the noise point, and at t equals 1 we are at the data point. *[Third formula.]* Differentiate with respect to t, and the velocity is x-one minus x-zero. It does not depend on t.

*[Figure.]* Here x-zero is minus 1.8 and x-one is 2, so the velocity is 3.8 at every time. *[Code line.]* This is the target in our five lines of code.

So now we have a velocity we can compute. But one line reaches only one data point, so this is not yet a generative model. The next step is to use many pairs.

### Formulas to explain

$$z = (x_0, x_1), \qquad x_t = (1 - t)\, x_0 + t\, x_1, \qquad u_t(x \mid z) = \frac{d x_t}{d t} = x_1 - x_0$$

| Term | What it means |
|:-----|:-----|
| z = (x₀, x₁) | One pair: a noise point and a data point |
| xₜ | The position on the line at time t |
| uₜ(x │ z) | The velocity of this line, given the pair |
| x₁ − x₀ | The slope of the line, the same at every t |

## Slide 10 · Wentao · 1.3 min · Step 2: mix over all pairs to get the path

### Script

Step two: use all the pairs. Draw many noise points and many data points, pair them independently, and draw every line. *[Left panel.]* Here are six lines. *[Middle panel.]* Here are two hundred. *[Right panel.]* If we keep adding pairs, the positions of the lines at each time form a smooth density. That density is p-t.

*[Formula.]* p-t of x is a mixture: p-t given z is the density on one line, and q of z says how pairs are drawn. With no extra noise, each line's density is just a point mass.

This is useful for two reasons. First, the endpoints are right by construction. At t equals 0, every line is at its noise point, so we get p-zero. At t equals 1, every line is at its data point, so we get the data distribution. Kaiming showed that many paths are possible. Now we have chosen one.

Second, we can sample p-t without any formula for it: draw a pair and a time, and compute x-t. That is the variable xt in our training loop.

So we have the path. What is its velocity field?

### Formulas to explain

$$p_t(x) = \int p_t(x \mid z)\, q(z)\, dz$$

| Term | What it means |
|:-----|:-----|
| pₜ(x) | The density at time t: this is the path |
| pₜ(x │ z) | The density on one line. With σ = 0 it is a point mass at xₜ |
| q(z) | How pairs are drawn, here independently |
| ∫ … dz | Mix over all pairs |

## Slide 11 · Wentao · 2 min · Step 3: the velocity is a posterior mean

### Script

This is the most important slide of the talk. Step three: what is the velocity of the mixture?

Stand at one point in space and time. Several lines pass through it, each with its own direction, but a velocity field can have only one value there. *[First formula.]* The theorem says that value is a posterior mean: the average of x-one minus x-zero, given that a line is at x at time t. *[Second formula.]* Written out, it is each line's velocity, weighted by the posterior probability of that line. That weight is Bayes' rule: prior q, likelihood p-t given z, divided by the evidence p-t.

Let's compute it at t equals 0.5, x equals 0.1. *[Figure.]* With two data points, exactly two lines pass through here. The one going to plus two must start at minus 1.8, so its velocity is plus 3.8. The one going to minus two must start at 2.2, so its velocity is minus 4.2. Which is more likely? Both data points have probability one half, so the noise density decides, and minus 1.8 is closer to zero than 2.2. The weights are 0.69 and 0.31. *[Table, bottom row.]* The average is plus 1.32. That single number is the velocity field at this point.

*[Why it helps generation.]* Why does this generate data? Each line, with its own velocity, satisfies the continuity equation. That equation is linear, so we can average it over q. The averaged flux is exactly p-t times this posterior mean. So p-t and u-t satisfy the continuity equation together, and by Kaiming's test, integrating u-t from noise gives data.

In general this integral is intractable, and the code never computes it. But our example is simple enough to check exactly.

### Formulas to explain

$$u_t(x) = E[\, x_1 - x_0 \mid x_t = x \,] = \int u_t(x \mid z)\, \frac{p_t(x \mid z)\, q(z)}{p_t(x)}\, dz$$

| Term | What it means |
|:-----|:-----|
| uₜ(x) | The velocity field we want |
| E[ · │ xₜ = x ] | The average over the lines through (t, x) |
| uₜ(x │ z) | The velocity of one line |
| pₜ(x │ z) | Likelihood: how strongly this pair puts a line at x |
| q(z) | Prior: how pairs are drawn |
| pₜ(x) | Evidence: it makes the weights sum to one |

The example at t = 0.5, x = 0.1, filled in one row at a time:

| Step | Line to +2 | Line to −2 |
|:-----|-----:|-----:|
| Start: x₀ = (x − t x₁) / (1 − t) | −1.8 | 2.2 |
| Velocity: x₁ − x₀ | +3.8 | −4.2 |
| Noise density: φ(x₀) | 0.0790 | 0.0355 |
| Posterior weight | 0.69 | 0.31 |

The number to remember: 0.69 × 3.8 + 0.31 × (−4.2) = 1.32.

The proof that the posterior mean generates data takes three lines. Each line satisfies the continuity equation; integrate over q; the flux becomes pₜ uₜ.

$$\partial_t p_t(x \mid z) + \nabla \cdot \big( p_t(x \mid z)\, u_t(x \mid z) \big) = 0$$

$$\partial_t p_t(x) + \nabla \cdot \int p_t(x \mid z)\, u_t(x \mid z)\, q(z)\, dz = 0$$

$$\partial_t p_t(x) + \nabla \cdot \big( p_t(x)\, u_t(x) \big) = 0$$

## Slide 12 · Wentao · 1.5 min · Step 4: check it, a velocity in closed form

### Script

Step four is a check. Our small example has a closed form, so we can test the claim without any network.

*[Box 1.]* First, rewrite the target. On a line, x-one minus x-zero equals x-one minus x-t, divided by one minus t: the distance still to go, over the time still left. Take the conditional expectation, and the velocity depends on one quantity: the expected data point given where we are now. This is a denoiser, and Kaiming will use it again for diffusion.

*[Box 2.]* Second, Bayes' rule. Given x-one, x-t is Gaussian with mean t x-one and standard deviation one minus t. With two data points, the posterior mean is a tanh, a soft sign of x.

*[Box 3.]* Put them together, and this is the exact velocity. At our point the denoiser gives 0.76, and 0.76 minus 0.1, over 0.5, is 1.32, as before.

*[Figure.]* Now the test. We integrated this formula from 200,000 Gaussian points. All reach plus or minus two, about half on each side. Nothing was trained.

Notice that the lines on the last slide cross, but these paths do not: a location and a time have only one velocity. This matters when we discuss why paths bend. On real data we cannot compute this mean, so we need a network.

### Formulas to explain

$$x_1 - x_0 = \frac{x_1 - x_t}{1 - t}, \qquad u_t(x) = \frac{E[x_1 \mid x_t = x] - x}{1 - t}$$

$$E[x_1 \mid x_t = x] = a \cdot \tanh \Big( \frac{a t x}{(1 - t)^2} \Big)$$

$$u_t(x) = \frac{1}{1 - t} \Big( a \cdot \tanh \Big( \frac{a t x}{(1 - t)^2} \Big) - x \Big)$$

| Term | What it means |
|:-----|:-----|
| (x₁ − xₜ) / (1 − t) | Remaining distance over remaining time |
| E[x₁ │ xₜ = x] | The denoiser: the best guess of the final data point from here |
| xₜ │ x₁ ~ N(t x₁, (1 − t)²) | The likelihood used in Bayes' rule |
| a · tanh( … ) | A soft sign: near 0 early on or when x is near 0, close to ±a late in time |

The number to remember: with a = 2, t = 0.5, x = 0.1, the velocity is (0.76 − 0.1) / 0.5 = 1.32.

## Slide 13 · Wentao · 1.7 min · Step 5: a random target has the same gradient

### Script

Step five answers our first question. We want the network to learn the velocity field, but we can compute only x-one minus x-zero for one pair. Given the input, that target is still random: at our point, it is plus 3.8 or minus 4.2. So why is regressing on it correct?

This is statistics you already know. *[First formula.]* This is the loss we can compute. *[Second formula.]* Call the target Y. Mean squared error splits into two terms: the squared distance from the network to the conditional mean of Y, plus the conditional variance of Y. The conditional mean is exactly u-t from step three, so the first term is the ideal loss. The second term does not depend on theta. *[Third formula.]* So the two losses differ by a constant and have the same gradient. Least squares on a noisy response recovers the regression function.

*[Figure.]* Here is our point. The horizontal axis is the network's output. The target is plus 3.8 with probability 0.69 and minus 4.2 with probability 0.31, and the network can output only one number. The expected loss is a parabola with its minimum at 1.32, at neither target.

*[Orange card.]* The height of that minimum, 13.69, is the conditional variance. So even a perfect network does not reach zero loss. In the tutorial, independent pairs end near 8 and optimal-transport pairs near 0.15. That mostly reflects how uncertain the target is, not sample quality. So never compare these models by loss value.

### Formulas to explain

$$L_{CFM}(\theta) = E_{t, z} \big\| v_\theta(x_t, t) - (x_1 - x_0) \big\|^2$$

$$E \| v_\theta - Y \|^2 = E \big\| v_\theta - E[Y \mid x_t, t] \big\|^2 + E\big[ \mathrm{Var}(Y \mid x_t, t) \big]$$

$$L_{CFM}(\theta) = L_{FM}(\theta) + C, \qquad \nabla_\theta L_{CFM} = \nabla_\theta L_{FM}$$

| Term | What it means |
|:-----|:-----|
| Y = x₁ − x₀ | The noisy response, which we can compute |
| E[Y │ xₜ, t] | The regression function, which is uₜ(x) from step 3 |
| E‖vθ − E[Y │ xₜ, t]‖² | The ideal loss L_FM from slide 8 |
| C = E[Var(Y │ xₜ, t)] | The target's variance. It does not depend on θ |

The number to remember: 0.69 × 0.31 × 8² = 13.69.

## Slide 14 · Wentao · 1.3 min · Step 6: from formulas back to code

### Script

Now we can go back to our five lines and label each one. *[Walk down the table.]* Draw a pair: that is z. Draw a time uniformly. Compute the point on the line: that is a sample from the path. Compute the target, x-one minus x-zero. Regress with mean squared error: that is the conditional flow matching loss. In torchcfm, one function does the time, the point and the target together. That is the long function name from our five lines.

*[Last row.]* Generation is the Euler loop from Kaiming's part: start from noise and add the network's velocity divided by K, K times.

*[Chain, left to right.]* This chain is the whole argument. We regress on a random target. By the variance decomposition, the network learns its posterior mean. By linearity, that mean satisfies the continuity equation with our path. And by the continuity equation, integrating it from noise generates data.

This answers our first two questions. The third remains: why would changing the pairs reduce the number of steps? If this already works, the only reason to change it is cost. So let's see what happens on 2D data with few steps.

### Formulas to explain

| Step | Formula | Code | From |
|:-----|:-----|:--------------|:-----|
| Draw a pair | z = (x₀, x₁) ~ q | `x1 = next(data_iter); x0 = torch.randn_like(x1)` | Slides 9–10 |
| Draw a time | t ~ U[0, 1] | `t = torch.rand(len(x1))` | Slide 8 |
| Sample the path | xₜ = (1 − t) x₀ + t x₁ | `xt = (1 - t) * x0 + t * x1` | Slide 10 |
| Compute the target | uₜ(x │ z) = x₁ − x₀ | `ut = x1 - x0` | Slide 9 |
| Regress | ‖vθ(xₜ, t) − uₜ(x │ z)‖² | `loss = torch.mean((model(t, xt) - ut) ** 2)` | Slide 13 |
| Generate | x ← x + (1/K) vθ(x, k/K) | `for k in range(K): x = x + model(k / K, x) / K` | Slide 6 |

Each arrow in the chain has one reason. Say it as you move along:

1. Regress on a random target → learn its posterior mean: the variance decomposition, slide 13.
2. Posterior mean → satisfies the continuity equation: linearity, slide 11.
3. Continuity equation → integrating generates data: the test from slide 7.

# Part III: slides 15–19

## Slide 15 · Wentao · 1 min · See it: 8 Gaussians to two moons

### Script

Now we train on a 2D dataset, using the experiment from the torchcfm tutorial. *[Top box.]* The source is eight Gaussians on a circle, the target is two moons, and the network is a small MLP. Training uses exactly the five lines we just labelled.

*[Let the animation run once.]* Three things to notice. First, it works: samples reach the moons, although the network only saw straight-line targets. Second, the source is not a standard Gaussian. We only had to sample from it. Third, look at the dark paths. They wander, and some loop back. On average, a path is 35 percent longer than the straight line from its start to its end.

*[Bottom card.]* That is strange, since every training target was straight. It also matters, because a solver must follow the curve, and each step costs one network call. So why do the paths bend?

### Formulas to explain

None on this slide. The number to remember: paths are 35 percent longer than the straight line.

## Slide 16 · Wentao · 1.4 min · Why paths bend, and what one step does

### Script

Back to our small example. *[Left panel.]* Here are the two lines through our point, with velocities plus 3.8 and minus 4.2. Independent pairs make lines cross everywhere, and at a crossing the network can give only the average.

*[Middle panel, left formula.]* Here is the consequence. Take a point starting at x equals 1. It will end at plus 2, but at time zero it knows nothing about its destination, so it aims at the mean, zero. Its velocity is minus 1, so it first moves away. As time passes, the posterior becomes sure, and the path turns. Lines cross, but marginal paths do not. Averaging is what bends them.

*[Right panel, right formula.]* This gives a prediction we can test. At time zero, x-zero is independent of x-one, so the velocity is the data mean minus x. One Euler step lands exactly on the mean, from every starting point. Here that is zero, where there is no data.

The W2 distance from the mean to the data is the square root of the trace of the covariance: 1.41 on our standardized data. We measured 1.28 to 1.37. So failure at one step is not a bug. This pairing implies it.

### Formulas to explain

$$u_t(x) = \frac{E[x_1 \mid x_t = x] - x}{1 - t}$$

$$u_0(x) = E[x_1] - x \quad \Rightarrow \quad x + u_0(x) = E[x_1]$$

| Term | What it means |
|:-----|:-----|
| E[x₁ │ xₜ = x] | Where the point expects to end. It is vague early and sure late, so the path turns |
| u₀(x) = E[x₁] − x | At t = 0, x₀ is independent of x₁, so every point aims at the data mean |
| x + u₀(x) = E[x₁] | One Euler step from t = 0 lands on the mean |
| W2 = √tr Cov = 1.41 | The predicted distance from the mean to standardized 2D data. Measured with I-CFM: 1.28–1.37 |

## Slide 17 · Wentao · 1.3 min · Change the pairs: optimal transport

### Script

If crossings cause the bending, let's stop them. *[Card 1.]* Did Part II use independence? It did not. The pair can follow any joint distribution with marginals p-zero and p-one, so the pairing is a free choice.

*[Card 2.]* Which pairing avoids crossings? Optimal transport: among all couplings, the one with the smallest expected squared length.

*[Card 3.]* Here is why. Take two lines crossing at time t, with velocities v and v-prime, and swap their endpoints. The total squared length falls by exactly two t, one minus t, times the squared velocity difference. That is positive, so an optimal coupling cannot contain a crossing.

*[Figure, top.]* Check it at our point. The crossing lines have squared lengths 3.8 squared and 4.2 squared, a total of 32.08. After the swap, each line moves only 0.2, a total of 0.08. The saving is 32, and the formula gives one half times 64, also 32. *[Figure, bottom.]* A batch of 16 points shows the same: mean squared length falls from 41 to 9.

*[Card 4.]* In code, it is one more line: solve the assignment on each batch, then continue as before. One warning: minibatch OT only approximates the true coupling.

### Formulas to explain

$$\pi^* = \underset{\pi \in \Pi(p_0, p_1)}{\arg\min}\; E_\pi \| x_1 - x_0 \|^2$$

$$2\, t\, (1 - t)\, \| v - v' \|^2 > 0$$

| Term | What it means |
|:-----|:-----|
| Π(p₀, p₁) | All joint distributions of (x₀, x₁) with the right marginals |
| π* | The pairing with the smallest expected squared length |
| v, v′ | The velocities of two lines that cross at time t |
| 2 t (1 − t) ‖v − v′‖² | How much squared length a swap of endpoints saves |

The number to remember: 2 · ½ · ½ · 8² = 32, and 32.08 − 0.08 = 32.

In code: `x0, x1 = ot_sampler.sample_plan(x0, x1)`.

## Slide 18 · Wentao · 1 min · Same data, straighter paths

### Script

Same data, same network, same starting points. Only that one line changed. *[Figure.]* I-CFM paths curve and loop. OT-CFM paths are almost straight.

*[Table, top two rows.]* Path length relative to the straight line falls from 1.35 to 1.001, and straightness from about 10 to 0.02. *[W2 row.]* W2 improves from 0.46 to 0.34.

*[Loss row.]* Remember the warning from Part II: the loss falls from about 8 to 0.15 because targets stop conflicting, not because the model is fifty times better.

*[Left card.]* Our small example shows the extreme case. With exact OT pairs, negative noise goes to minus 2 and positive noise to plus 2. Each point has one target, the loss floor falls from 13.69 to zero, and one Euler step is exact.

*[Last row.]* It is not free: 5,000 training steps take about 120 seconds instead of 35 to 61. Does straighter really mean fewer steps?

### Formulas to explain

$$S = E \| v - (x_1 - x_0) \|^2$$

| Term | What it means |
|:-----|:-----|
| S | Straightness: how far the velocity is from the straight-line velocity along the path. 0 is best |
| x₁ − x₀ | The start-to-end displacement of a generated path. S is measured over 200 steps |
| Length / straight line | The path length divided by the straight distance. 1 means straight |

## Slide 19 · Wentao · 1.3 min · The payoff: good samples in few steps

### Script

This is our own experiment, on four datasets, with a fixed number of solver steps. The source is now a standard Gaussian and the data are standardized, so do not compare with the last table. Each curve is one run, seed 0. Lower is better, and the dashed line is the distance between two sets of real samples.

*[Orange circles.]* OT-CFM is nearly flat from the first step, so in 2D one to five steps are enough. *[Brown squares.]* I-CFM starts above 1, as we predicted, and comes close only at 20 to 50 steps. On 8 Gaussians, it stays at 0.31, above OT-CFM at 0.20.

*[Blue triangles.]* DDIM, a fast sampler for diffusion models, is worst at few steps. *[Diamond.]* This is DDPM with 1,000 steps. *[Pictures.]* At 5 steps, both flow models keep the moons, and DDIM does not. Kaiming explains why next.

*[Last line.]* Rectified flow is another way to straighten: retrain on the pairs the model itself generates. Details are in the backup.

This answers question 3 in 2D: pairs that do not cross give straight paths, and straight paths need few steps. Kaiming, over to you.

### Formulas to explain

None on this slide. Explain the axes: the horizontal axis is NFE, the number of solver steps, and the vertical axis is W2 to a test set, where lower is better.

# Part IV: slides 20–22

## Slide 20 · Kaiming · 1.2 min · DDPM: the same regression on another path

### Script

Wentao built flow matching from straight lines. Diffusion fits the same template. *[Formula.]* Every path is a weight on data times x-one, plus a weight on noise times x-zero. *[Table.]* For flow matching, the weights are t and one minus t. For DDPM, they are the square roots of alpha bar and one minus alpha bar, so their squares add to one: variance preserving.

*[Notation card.]* One warning. DDPM runs time backward, calls the data x-zero, and calls the noise epsilon. In this talk, t equals 0 is noise, t equals 1 is data, and x-zero is the noise.

*[Left plot.]* Here are the weights. On VP, the data weight stays below 0.3 until halfway. *[Other plots.]* Here are exact paths for our example, with nothing trained. On VP they stay nearly flat, then move late.

*[Bottom card.]* Training is the same idea. DDPM regresses on the noise instead of the velocity, and by the argument from Part II, the best network is again a posterior mean. So how does that relate to the velocity?

### Formulas to explain

$$x_t = a_t\, x_1 + b_t\, x_0$$

$$\min_\theta E \| \epsilon_\theta(x_t, t) - x_0 \|^2 \quad \to \quad \epsilon_\theta^*(x, t) = E[x_0 \mid x_t = x]$$

| Term | What it means |
|:-----|:-----|
| aₜ, bₜ | The weights on data and on noise. Straight path: t and 1 − t |
| √ᾱ, √(1 − ᾱ) | The DDPM weights, called VP. aₜ² + bₜ² = 1 |
| ᾱ = ∏(1 − βᵢ) | β is linear from 10⁻⁴ to 0.02 over 1,000 steps. At t = 0, a = 0.0064 |
| εθ | The network that predicts the noise |
| E[x₀ │ xₜ = x] | The best noise network is a posterior mean |

DDPM's notation in ours: its x₀ is our data x₁, its ε is our noise x₀, its step 1,000 is our t = 0, and its step 0 is our t = 1.

## Slide 21 · Kaiming · 1.2 min · Velocity, noise, score: one posterior mean

### Script

*[Card 1.]* Take the path equation and condition on x-t equals x. That gives one linear equation linking the expected data and the expected noise. Know one, and you know the other.

*[Card 2.]* The velocity is their difference. Substituting gives x minus epsilon-hat, over t. So a noise network gives a velocity, with no retraining.

*[Card 3.]* For the score, use Fisher's identity: the score of a mixture is the posterior average of the component scores. Each component is Gaussian, with score minus x-zero over one minus t. Averaging gives minus epsilon-hat over one minus t.

*[Orange card.]* Check it on our point. The expected data is 0.76, as on slide 12. The expected noise is minus 0.56. The velocity is 1.32 again, and the score is 1.12, which the mixture confirms directly.

*[Where each step holds.]* Steps one and two are algebra, so they always hold. The score step needs x-zero given x-one to be Gaussian, so it fails for a mixture source and for OT pairs.

All three share one optimum, but a network's errors do not convert evenly. That matters next.

### Formulas to explain

$$x = t\, \hat{x}_1 + (1 - t)\, \hat{\epsilon}$$

$$u_t(x) = E[x_1 - x_0 \mid x_t = x] = \hat{x}_1 - \hat{\epsilon} = \frac{x - \hat{\epsilon}}{t}$$

$$\nabla \log p_t(x \mid x_1) = -\frac{x - t x_1}{(1 - t)^2} = -\frac{x_0}{1 - t}$$

$$s_t(x) = \nabla \log p_t(x) = -\frac{\hat{\epsilon}}{1 - t}$$

| Term | What it means | At t = 0.5, x = 0.1 |
|:-----|:-----|-----:|
| x̂₁ = E[x₁ │ xₜ = x] | The expected data point | 0.76 |
| ε̂ = E[x₀ │ xₜ = x] | The expected noise | −0.56 |
| uₜ(x) = (x − ε̂) / t | The velocity | 1.32 |
| sₜ(x) = −ε̂ / (1 − t) | The score | 1.12 |

Direct check from the mixture N(±1, 0.25): 0.69 · 3.6 − 0.31 · 4.4 = 1.12.

## Slide 22 · Kaiming · 1.4 min · DDIM is this ODE. So why more steps?

### Script

*[Top card.]* One DDIM step uses the noise network to estimate both ends, then re-mixes them at the next time. On the straight path, this is exactly one Euler step. On VP, divide x by a plus b, use a over a plus b as the clock, and it is the same Euler step. *[Figure, top.]* The orange VP paths fall on the straight ones.

*[Orange card.]* So the gap on slide 19 is not about path shape. *[Table, NFE 1.]* At one or two steps, the cause is dividing by a, which is 0.0064 at the first call. A noise error of 0.05 to 0.09 becomes 8 to 14, close to the measured W2. With the exact field on our example, one step gives 1.98, with no blow-up.

*[Table, NFE 5.]* At five steps, it is mostly the grid: the same velocity network goes from about 0.3 to about 0.5 on DDIM's grid. This depends on the data: on our two-point example, DDIM's grid does better. We have not separated what remains at 20 steps.

*[DDPM card.]* DDPM keeps the same marginals and adds noise. In the continuity equation, the extra drift and the noise cancel.

Only the pairing changes the path. Next, images.

### Formulas to explain

$$\hat{x}_1 = \frac{x - b\, \hat{\epsilon}}{a}, \qquad x' = a'\, \hat{x}_1 + b'\, \hat{\epsilon}$$

$$dx = \big( u + \tfrac{1}{2} g^2 s \big)\, dt + g\, dW$$

| Term | What it means |
|:-----|:-----|
| x̂₁ = (x − b ε̂) / a | Estimate the data end from the noise network. At the first call b / a = 157, so noise error is magnified |
| x′ = a′ x̂₁ + b′ ε̂ | Re-mix both ends at the next time. With a = t, b = 1 − t this is exactly Euler |
| ½ g² s | The extra drift DDPM adds, using the score s |
| g dW | The added noise. It cancels the extra drift in the continuity equation, so the marginals stay the same |

# Part V: slides 23–28

## Slide 23 · Kaiming · 1 min · Step 1: what changes from 2D to images

### Script

So far everything lived in two dimensions. Now we move to images, and the first question is what has to change.

*[Table.]* The data go from 2 numbers to 784 pixels. The MLP becomes a U-Net with about a million weights, and training takes half an hour per model instead of a few minutes.

*[Orange card.]* But these are the same lines as on slide 4. That is not luck: the derivation in Part II never used the dimension.

*[White card.]* One thing must change: the metric. Empirical W2 converges like n to the minus one over d, which is hopeless here. So Wentao will use FID on slide 26.

*[Bottom row, left to right.]* We carry three tests from 2D. I-CFM in one step should land on the data mean. OT-CFM should need fewer steps. DDIM should be weakest at very few steps. Let's see which survive.

### Formulas to explain

$$W_2(\hat{p}_n, p) \sim n^{-1/d}$$

| Term | What it means |
|:-----|:-----|
| n | Number of samples: 2,000 in 2D |
| d | Dimension: 2, then 784 |
| n^(−1/d) | How fast the empirical distance settles. To halve the error, you need about 2^d times more samples, which is why W2 is hopeless at d = 784 and slide 26 uses FID |

The code card shows the lines from slide 4. With σ = 0, the library call `fm.sample_location_and_conditional_flow(x0, x1)` returns exactly t, xₜ and x₁ − x₀.

## Slide 24 · Kaiming · 1.1 min · Step 2: what the network sees

### Script

*[Figure, one row.]* Each row is one pair: noise on the left, a digit on the right, and the straight-line mix in between.

*[Last column.]* The target, x-one minus x-zero, is mostly noise with a faint digit. No network can match it exactly, so the loss never reaches zero, as on slide 13.

*[Bottom card.]* Input and output have the same shape, so a U-Net is natural. Time enters each residual block through an embedding.

*[Orange card.]* The best output is still the posterior mean, now in 784 coordinates. Two pixels make that concrete.

*[Corner card.]* The corner pixel is minus 1 in every training image, so x-one is known. The velocity is minus 1 minus x, over one minus t, and the path is straight.

*[Center card.]* The center pixel is stroke or background. At first the network can only aim at the mean. Later the neighbors reveal the digit, the aim changes, and the path bends. This is slide 16, one pixel at a time.

### Formulas to explain

$$u_t(x) = E[\, x_1 - x_0 \mid x_t = x \,], \qquad u_t^{\text{corner}}(x) = \frac{-1 - x}{1 - t}$$

| Term | What it means |
|:-----|:-----|
| x | The whole image at time t: 784 numbers |
| E[· │ xₜ = x] | The average over all pairs that could produce this image. It gives one vector, one velocity per pixel |
| −1 | The corner pixel's value in all 60,000 training images, so x₁ is known at that pixel. 67 of the 784 pixels behave this way |
| 1 − t | The time left: remaining distance over remaining time |

How to derive the corner formula: since xₜ = t x₁ + (1 − t) x₀, we have x₁ − x₀ = (x₁ − xₜ) / (1 − t), and x₁ = −1 leaves no uncertainty. The center pixel has mean 0.02 and standard deviation 0.89, so its posterior changes as the neighbors reveal the digit. That change bends the path.

## Slide 25 · Kaiming · 1.2 min · Step 3: from noise to digits, with the same noise

### Script

*[Grid.]* Every panel starts from the same noise. Rows are methods, columns are network calls. *[Bottom strip.]* This strip shows OT-CFM over time: digits appear by about t equals 0.5.

*[First column, top row.]* I-CFM at one call gives the same gray blur in every cell. *[Top right.]* Compare it with the data mean: it is the same picture. That is test 1, and it holds at 784 dimensions. *[Table.]* After one step, I-CFM is 0.04 from the mean, while a real digit is 0.52 away.

*[First column, middle row.]* OT-CFM at one call is different. It is blurry, because one step lands on the mean of the digits paired with that noise, but you can already read digits, and different noise gives different digits.

*[First column, bottom row.]* DDIM at one or two calls gives white specks. About three quarters of the pixels leave the valid range, with values up to 19. That is the division by a small a from slide 22.

*[Last columns.]* By 10 calls all three look like digits, and pictures stop being a fine enough test. Wentao, over to you.

### Formulas to explain

$$x_0 + u_0(x_0) = E[\, x_1 \mid x_0 \,] \quad \overset{\text{independent pairs}}{\longrightarrow} \quad E[\, x_1 \,]$$

| Term | What it means |
|:-----|:-----|
| x₀ | The starting noise: 784 numbers |
| u₀(x₀) | The velocity at t = 0, where xₜ = x₀. It equals E[x₁ │ x₀] − x₀ |
| E[x₁ │ x₀] | The mean of the digits paired with this noise. It depends on x₀ only if the pairs are not independent |
| E[x₁] | The data mean: the blur at the top right of the slide |

How to read the numbers (RMS per pixel, pixels in [−1, 1]; a real digit is 0.52 from the mean):

| After one step | I-CFM | OT-CFM |
|:-----|:-----|:-----|
| Distance to the data mean | 0.04 | 0.22 |
| Spread across noise | 0.04 | 0.22 |
| To nearest training image | 0.39 | 0.35 |

## Slide 26 · Wentao · 1.4 min · Step 4: quality vs. steps, measured on two seeds

### Script

Now the measurement. *[Top card.]* FID fits a Gaussian to Inception features of real images and of generated images, then compares their means and covariances. Lower is better, and two sets of real images give 0.54. Inception was trained on ImageNet, so on digits we only read trends.

*[Plot.]* We trained every model twice, with two seeds. Solid lines are seed 0, dashed lines seed 1.

*[Orange card.]* At one and two steps OT-CFM is lowest on both seeds, and I-CFM is 1.5 to 1.7 times higher. That part of test 2 holds. From 10 steps on, the seeds disagree: on seed 1, I-CFM is lower.

*[Blue curves.]* DDIM is worst from 2 to 20 steps on both seeds, as test 3 predicted. At 100 steps it is the best model on seed 0, 12.2, and the worst on seed 1, 32.8. Same code and same data; only the seed changed.

*[Bottom card.]* Two cautions. At one step the order of DDIM and I-CFM flips between seeds, so do not read FID at that scale. And our best value, 11.5, is far above the floor: this is demo-level training.

So OT-CFM helps when steps are very few. How much of the 2D advantage is left?

### Formulas to explain

$$\mathrm{FID} = \| \mu - \mu' \|^2 + \mathrm{tr}\big( \Sigma + \Sigma' - 2 (\Sigma \Sigma')^{1/2} \big)$$

| Term | What it means |
|:-----|:-----|
| μ, Σ | Mean and covariance of Inception features of real images: 60,000 training images |
| μ′, Σ′ | The same for generated images: 10,000 samples |
| First term | How far apart the two feature means are. It sees a wrong average |
| Second term | How different the two spreads are. It sees too little or too much variety |
| FID | Squared W2 between the two fitted Gaussians. Lower is better; real vs. real gives 0.54 |

How to read the results: use thresholds, because they answer how many calls are needed. Each cell is seed 0 / seed 1; a dash means the model does not get there within 100 calls.

| Calls needed for | OT-CFM | I-CFM | DDIM |
|:-----|:-----|:-----|:-----|
| FID below 30 | 5 / 10 | 10 / 10 | 20 / — |
| FID below 20 | 10 / 50 | 50 / 20 | 100 / — |

The two training runs of DDIM end with almost the same loss, 0.0215 and 0.0213, but their FID at 100 steps is 12.2 and 32.8. The loss does not tell you which run is better.

## Slide 27 · Wentao · 1.2 min · Step 5: does the OT advantage survive at d = 784?

### Script

*[Table header.]* W2 in 2D and FID on MNIST are different scales, so read down each column, not across.

*[First two rows.]* In 2D, OT pairing cuts the cost by about 90 percent. On MNIST the ratio is 0.96. *[Middle card.]* But only the inner product can change. The most any pairing can save is 386. A batch of 128 saves 66, or 17 percent of that, and a batch of 2,048 saves only 88. So minibatch pairs stay closer to random than to exact OT.

*[Orange card.]* We expected that to erase the advantage, and with two seeds it nearly does. At one and two steps OT-CFM's FID is still 33 to 42 percent lower on both seeds. At five steps the gap is 41 percent on seed 0 but only 3 percent on seed 1, and from ten steps on the seeds disagree.

*[Bottom right.]* Path length hides this: 1.02 looks straight, because most of the length only removes noise.

So our hypothesis is largely supported. But FID is only one score. Kaiming will now ask a different question, with a course method.

### Formulas to explain

$$E\| x_0 - x_1 \|^2 = E\| x_0 \|^2 + E\| x_1 \|^2 - 2 E\langle x_0, x_1 \rangle$$

$$\text{largest possible saving} = 2\, \mathrm{tr}\, \Sigma^{1/2}$$

| Term | What it means |
|:-----|:-----|
| E‖x₀‖² | Noise size, fixed by the source: 784 |
| E‖x₁‖² | Data size, fixed by the data: 725 |
| −2 E⟨x₀, x₁⟩ | The only term a pairing can change. It is 0 for random pairs |
| Σ | Covariance of the data: tr Σ = 211 |
| 2 tr Σ^(1/2) | The most any coupling can save (Gelbrich bound): 386 |

Read down each column of the table, not across, because W2 and FID are on different scales. MNIST cells are seed 0 / seed 1; the pair and path rows use seed 0.

| | 2D (four datasets) | MNIST |
|:-----|:-----|:-----|
| Pair cost, OT / random | 0.07–0.13 | 0.96 |
| Share of the largest saving | 85–94% | 17% |
| Path / line, I-CFM → OT-CFM | 1.8–2.5 → 1.01 | 1.02 → 1.01 |
| I-CFM / OT-CFM at 1 step | 6.6–10.8× | 1.5× / 1.5× |
| I-CFM / OT-CFM at 5 steps | 1.5–3.3× | 1.7× / 1.0× |
| I-CFM / OT-CFM at 100 steps | 1.1–1.5× | 1.04× / 0.77× |
| OT-CFM, 1 step / 100 steps | 1.0–1.3× | 16× / 12× |

## Slide 28 · Kaiming · 1.7 min · E3: are all kinds of digits generated in the right shares?

### Script

FID gives one number, from ImageNet features. Our course method asks a different question: does the generator make every kind of digit, in the right proportion?

*[Bottom card.]* We fit k-means with ten clusters on 30,000 real training images only, so the clusters do not depend on any model. Each generated image goes to its nearest real center. Then we compare the shares with the real ones. TV is half the sum of the absolute differences: the share of images that would have to move to another cluster. With ten thousand images the chi-squared test rejects for every model, so we read the size of the gap: TV, and the effect size w.

*[Left panel.]* Solid lines are seed 0, dashed lines the second training run. On seed 0, DDIM in blue is far the worst, at 0.19. *[Right panel.]* It makes 19 percent ones and only four and a half percent threes. And this is the model with the best FID at 100 steps.

*[Orange card.]* DDIM's second training run is even, at 0.06, like the others, while its FID got worse. So the imbalance came from one training run, not from the method, and FID and shares measure different things.

*[Middle card.]* Two things hold on both seeds: every model stays 2.6 to 11 times above the real baseline, and shares stop improving after about 20 steps. *[Last card.]* And two limits: clusters are not digits, and at one or two steps blurry images sit near the centers, so we do not read those.

### Formulas to explain

$$\mathrm{TV} = \tfrac{1}{2} \sum_{k=1}^{10} \lvert \hat p_k - p_k \rvert$$

$$\chi^2 = \sum_{k=1}^{10} \frac{(O_k - E_k)^2}{E_k}, \qquad w = \sqrt{\chi^2 / n}$$

| Term | What it means |
|:-----|:-----|
| p_k | Share of real images in cluster k, from 30,000 held-out training images |
| p̂_k | Share of the n = 10,000 generated images in cluster k |
| TV | The fraction of generated images that would have to change cluster to match the real shares. 0 is a perfect match. Real test set: 0.017 |
| O_k, E_k | Observed count in cluster k, and the expected count n p_k |
| χ² | Pearson's test of the shares. With n = 10,000 it rejects for every model from 5 steps on (p < 10⁻¹⁸), so the p-value says little |
| w | Effect size. It does not grow with n: about 0.1 is small and 0.3 medium (Cohen). Real test set: 0.04; models at 100 steps: 0.11–0.46 |

How to explain the example: TV 0.19 means that about one in five of DDIM's seed-0 images would have to change cluster to match the real proportions. For OT-CFM seed 0 it is about one in twenty-three.

At 100 steps (TV, mean of three noise draws):

| | Seed 0 | Seed 1 |
|:-----|:-----|:-----|
| OT-CFM | 0.044 | 0.070 |
| I-CFM | 0.088 | 0.055 |
| DDIM | 0.190 | 0.060 |
| Real test set | 0.017 | |

New noise moves TV by at most 0.012; the training seed moves it by up to 0.13. If someone asks why k-means and not a classifier, see slide 29. If someone asks about the digit labels: they come from an RBF-SVM trained on 10,000 real images, with 97% test accuracy, and are used only to say which digits are missing.

# Close: slides 29–31

## Slide 29 · Kaiming · 1 min · Project plan: what is done, what comes next

### Script

This is where the project stands. *[Top table.]* E1, in 2D, is done for one seed. E2 and E3, on MNIST, are done for two training seeds per method.

*[Lower table.]* The course asks for one course method, and this table shows why we chose k-means. In a pilot, train on fake and test on real gave about 90 percent for every model, so it could not tell them apart. A real-versus-fake classifier was right almost every time, mostly because generated backgrounds are not exactly black. PCA with a Fréchet distance worked, but it is one more score like FID. k-means shows which kinds of digits are missing or too frequent.

*[Orange card.]* Next: more seeds, since two already disagree, then DDPM with 1,000 steps and Fashion-MNIST. Wentao will close.

### Formulas to explain

None on this slide. The pilot used the seed-0 models and 5,000 samples per model, so call its numbers a pilot.

## Slide 30 · Wentao · 0.8 min · Takeaways

### Script

Five things to remember. *[1.]* Flow matching turns generation into regression: least squares gives a posterior mean, and that is a correct velocity field.

*[2.]* Pairing sets the path, and the path sets the number of steps.

*[3.]* Velocity, noise and score are one quantity. Few-step gaps come from conversions and grids, not from path shape.

*[4.]* In 784 dimensions the OT advantage is left only where steps are very few: 33 to 42 percent lower FID at one and two steps, on both seeds.

*[5.]* Check more than FID, and more than one seed. Our course method found that the best-FID model was the most unbalanced, and a second training run did not repeat it.

### Formulas to explain

None on this slide. Line 1 answers Part II, line 2 Part III, line 3 Part IV, and lines 4 and 5 Part V. Do not add new material here.

## Slide 31 · Wentao · 0.2 min · Questions

### Script

That is the talk. Thank you, and we are happy to take questions. The backup slides have the proofs and the metric definitions.

### Formulas to explain

None on this slide. Leave the references on screen during questions.

# Backup: slides B1–B6

These slides are not timed and have no script. Open one only if a question needs it.

| Slide | Use it when someone asks | The one line to say |
|:-----|:-----|:-----|
| B1 | "Prove that the marginal field gives the right density" (slides 7, 11) | With a test function, it is one use of the tower rule |
| B2 | "Show L_CFM = L_FM + C" (slide 13) | C is an expected conditional variance, so it holds no θ |
| B3 | "How does this relate to Lipman's paper?" | With σ_min = 0 and a Gaussian source, the regression is the same |
| B4 | "What is rectified flow?" (slide 19) | Retrain on the model's own pairs. We have not run it |
| B5 | "What about Schrödinger bridges?" | W2 0.37 ± 0.05 overlaps OT-CFM at 0.34 ± 0.06, so we do not rank them |
| B6 | "What exactly are W2, MMD and FID?" (slides 18, 19, 26) | FID is W2² between Gaussians fitted to Inception features |
