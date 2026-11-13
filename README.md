# Anticipating synchronization from aggregate data

Can a machine trained only on data from *before* a synchronization transition predict where it happens, when it sees only the network's average signal? This project reproduces a published reservoir-computing result, tests it on a 50-node chaotic network observed through its mean field, and follows the failures to a physical explanation.

**Short answer.** A parameter-aware reservoir trained on one simulation per coupling value cannot anticipate complete synchronization from the mean field. The reason is physical: below the threshold the network flickers in and out of synchrony (on-off intermittency), so runs at the same coupling differ wildly. The mean field still reads each run's hidden state almost perfectly (per-run correlation up to 0.98). Training the reservoir on three or more runs per coupling fixes most of the failure, and it then tracks the approach to synchronization reproducibly for about 0.02 to 0.04 in coupling beyond its training data.

---

## 1. Background

Fan, Kong, Lai and Wang (2021), *Anticipating synchronization with machine learning*, Phys. Rev. Research 3, 023237 ([arXiv:2103.13358](https://arxiv.org/abs/2103.13358)), feed a reservoir computer the control parameter ε as an extra input, train it on a few desynchronized states, and sweep ε in closed loop to predict the onset of synchronization. Their systems are small (two coupled maps or oscillators, a 4-node star), the reservoir sees every node's full state, and each figure shows a single reservoir. Their discussion names large systems as an open problem.

Two questions follow:

1. How reliable is the method across random reservoirs?
2. Does it work on a larger network when only an aggregate signal is observed, as with EEG?

## 2. Reproduction: two coupled logistic maps

**Setup.** x₁′ = F(x₁) + ε[F(x₂) − F(x₁)] and symmetrically, with F(x) = 4x(1 − x). Complete synchronization begins at ε₁ ≈ 0.25. Reservoir: 100 neurons, spectral radius 1e-5, ridge λ = 1e-5, trained on ε = 0.20, 0.22, 0.24.

**Result.** A single reservoir reproduces Fan et al.'s Fig. 2(a): the predicted synchronization error falls to about 0 at ε ≈ 0.25.

**Reliability across reservoirs.** Over 30 to 50 random reservoirs, each one either works (error about 0.002) or fails completely (outputs down to −20). The failure rate grows with distance past the training range:

| ε | 0.25 | 0.255 | 0.26 | 0.27 |
|---|---|---|---|---|
| reservoirs that fail | 0% | ~3% | 6–10% | 10–17% |

**Mechanism.** The logistic map at r = 4 maps [0, 1] exactly onto itself with no margin: F(0.5) = 1 and anything above 1 maps below 0, after which the orbit runs to −∞. Before synchronization the coupled map averages two values and rarely reaches the edges, so the training data rarely constrains the learned map there. After synchronization each map is the plain F, whose orbit spends much of its time near 0 and 1. A reservoir whose learned peak is 1.0003 instead of 1 escapes; failing reservoirs stay synchronized while doing so.

**Accuracy versus reliability.** Training further from the transition (ε = 0.18 to 0.22) gave 0 of 50 failures, because the learned peak sat about 0.3% below 1 (median 0.9974), a safety margin created by sparse edge data. Training closer (0.20 to 0.24) fitted the peak almost exactly (median 0.9995) and failed in 5 of 50. Other settings, at ε = 0.27:

| change | failure rate | cost |
|---|---|---|
| baseline (λ = 1e-5, 100 neurons) | 10% | none |
| weaker ridge, λ = 1e-7 | 26% | none |
| stronger ridge, λ = 1e-3 | 0% | error floor about 10× higher |
| 300 neurons | 4% | none; more accurate |
| clip outputs to the training range | 0% | relies on the system living in a box |

These rates come from 50 reservoirs per setting and should be confirmed with more.

## 3. The network: 50 Rössler oscillators

**System.** Identical Rössler oscillators (a = b = 0.2, c = 5.7), coupled diffusively through x on a Watts–Strogatz graph (M = 50, k = 6, p = 0.15). RK4 integration, dt = 0.02, transient of 200 to 300 time units discarded.

```
dx_i/dt = -y_i - z_i - ε Σ_j L_ij x_j
dy_i/dt =  x_i + a y_i
dz_i/dt =  b + z_i (x_i - c)
```

**Master stability function.** The largest Lyapunov exponent of the variational equation with coupling matrix diag(1, 0, 0) is negative for 0.143 < σ < 4.440. With Laplacian eigenvalues λ₂ = 0.819 and λ_N = 10.826, complete synchronization is stable for **0.175 < ε < 0.410**.

**Simulation (10 runs per ε).**

- Phase synchronization comes early: the Kuramoto order parameter R rises from 0.14 at ε = 0 to 0.98 by ε = 0.04.
- Complete synchronization matches the MSF window: the synchronization error falls to 0.039 at ε = 0.22, reaches 0 by 0.32, and returns at 0.40 (0.054) and 0.42 (0.506). The lag behind 0.175 reflects slow convergence near the threshold.
- At ε ≥ 0.44 trajectories leave the Rössler basin and diverge.
- With a 1% frequency mismatch the error levels off at 0.10 to 0.15 instead of 0.

## 4. Predicting from the mean field

**Observation.** The reservoir sees only the mean field of the 50 oscillators (3 numbers instead of 150). Complete synchronization is detected through the standard deviation of the mean-field z: under complete synchrony the oscillators spike together and the average reproduces a full spike.

**Reservoir.** 600 neurons, spectral radius 0.9, leak 0.5, density 0.05, input scale 0.3, parameter-channel scale 0.2, ε scaled as 5ε + bias, ridge 1e-4, training-input noise 0.02. Trained on ε = 0.10, 0.12, 0.14, 0.16, then run in closed loop for 3000 steps after a 300-step warm-up. Predictions are labelled blowup, collapse (z std < 0.3) or runaway (above 1.5 × the truth).

**Findings, in the order they were tested.**

1. *Apparent success was luck.* With 20 reservoirs, two disjoint halves of 10 disagreed: medians of 2.83 and 2.40 at ε = 0.20 (truth 2.77). Individual predictions at 0.20 ranged from 0.76 to 6.39.
2. *Bigger reservoirs made it worse.* No size from 600 to 3000 neurons captured the rise at ε = 0.18 (medians 2.34 to 2.43 against 2.68). The fraction of sensible predictions at ε = 0.30 fell from 0.55 to 0.20 as size grew.
3. *A hand-tuned setting looked good but did not survive a fair test.* A bias of −0.25, picked by inspecting the test curve, appeared to work. With fresh simulations, a fixed bias rule and three training windows (0.06–0.10, 0.08–0.12, 0.10–0.14), no window anticipated the sustained level after the transition.

## 5. Why: on-off intermittency below the threshold

Plotting the synchronization error over time at ε = 0.16 shows the network switching between long nearly-synchronized stretches and bursts of desynchrony. The fraction of time with error below 1.0 (20 runs per ε):

| ε | 0.04 | 0.08 | 0.10 | 0.11 | 0.12 | 0.13 | 0.14 | 0.16 | 0.17 |
|---|---|---|---|---|---|---|---|---|---|
| time near sync | 0.00 | 0.05 | 0.11 | 0.18 | 0.24 | 0.31 | 0.39 | 0.59 | 0.68 |

Intermittency grows steadily from ε ≈ 0.08, well below the MSF threshold of 0.175. Runs at the same ε differ greatly (standard deviations of 0.2 to 0.4), so ε controls the network's state only loosely.

**The mean field reads each run's state.** Correlation across 20 runs, at fixed ε, between the run's near-sync time and a mean-field-only statistic:

| ε | 0.08 | 0.10 | 0.11 | 0.12 | 0.13 | 0.14 | 0.16 | 0.17 |
|---|---|---|---|---|---|---|---|---|
| std of mean-field z | 0.85 | 0.87 | 0.95 | 0.96 | 0.98 | 0.97 | 0.97 | 0.90 |
| fraction of z spikes above 12 | 0.83 | 0.81 | 0.90 | 0.94 | 0.96 | 0.94 | 0.90 | 0.93 |

The run-averaged z standard deviation looked flat across 0.10 to 0.16 (2.33, 2.24, 2.41, 2.31) because averaging mixes strongly and weakly intermittent runs. Per run, the signal is strong. The tall-spike count is no better than the plain standard deviation, so the claim is that the mean field carries the state, not that one statistic is special.

## 6. The fix: several runs per coupling

If the failure comes from the loose link between ε and the state, a reservoir trained on one run per ε learns mostly which random run it happened to see. Training on several runs per ε should average that out. Same training window (0.10 to 0.16), fixed bias rule, fresh test simulations, 20 reservoirs:

| ε | 0.14 | 0.16 | 0.18 | 0.20 | 0.22 |
|---|---|---|---|---|---|
| truth | 2.59 | 2.72 | 2.85 | 2.78 | 2.89 |
| 1 run per ε | 2.06 | 2.94 | 3.25 | 2.94 | 1.10 |
| 3 runs per ε | 2.40 | 2.54 | 2.69 | 2.97 | 2.38 |
| 5 runs per ε | 2.36 | 2.50 | 2.65 | 2.75 | 2.27 |

With one run per ε the reservoir is wrong even at ε = 0.14, a value it trained on, then overshoots and collapses, and its halves disagree. Three runs per ε give a smooth rise whose halves agree through ε = 0.18 (2.75 and 2.68). Five runs add nothing over three. All versions underpredict by about 0.2 in range and fail by ε = 0.22.

## 7. Conclusions

1. Fan et al.'s logistic-map result reproduces, but 10 to 17% of random reservoirs blow up past the transition because the synchronized map has no margin at its edges. A single reservoir is not evidence.
2. On a 50-node Rössler network observed through its mean field, a reservoir trained on one run per coupling cannot anticipate complete synchronization. Apparent successes came from luck or tuning.
3. Below the stability threshold the network shows on-off intermittency, growing from ε ≈ 0.08, with large run-to-run variation.
4. Simple mean-field statistics track each run's hidden near-sync time (r = 0.81 to 0.98). The information is in the aggregate signal; what failed was learning a clean dependence on ε from single runs.
5. Training on three or more runs per coupling fixes most of this, giving reproducible predictions about 0.02 to 0.04 past the training data.

## 8. Limitations and next steps

- One network topology, identical oscillators, no measurement noise.
- 20 reservoirs per setting; the runs-per-coupling result uses one set of training seeds and should be repeated with another.
- Thresholds (near-sync error below 1.0, spikes above 12) were chosen by hand; 8, 10 and 12 all gave the same trend.
- The negative correlation between near-sync time and mean-field x spread is unexplained.

Next: partial observation (mean field plus a few individual nodes), other topologies (scale-free), neuron models, and testing the mean-field indicators on EEG recordings.

