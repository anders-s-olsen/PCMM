# Directional audio mixture experiment

This experiment asks whether independently fitted spatial mixture models recover
three talker directions in a measured car acoustic environment, and how that
answer changes when measured driving noise is added. The three sources are
different MiniLibriMix speakers. They occupy McVAMPIRE mouth positions **1, 2,
and 3**: the two front seats and one rear seat. A fourth measured position,
**seat 5**, is empty and is retained as a negative-control support point in the
spatial visualization.

The two data conditions are paired:

- `no_driving_noise` contains the rendered three-speaker conversation alone.
- `car_noise` contains the exact same rendered source scene plus measured
  multichannel McVAMPIRE driving noise.

Thus speech identity, utterances, activity, seat responses, train/test split,
STFT grid, and source gains are identical across conditions. Only the added
driving noise differs. There is no reserved background component; every value
of K is the total number of unconstrained mixture components.

## Run it

Use Python 3.10 or later. Install the requirements, then edit the corpus paths
in `config.json` to point to extracted MiniLibriMix and In-Car McVAMPIRE trees.
Relative paths are resolved from the configuration file.

```bash
python -m pip install -r requirements.txt
python experiment.py prepare --config config.json
python experiment.py run --config config.json
```

`prepare` writes the paired observations, source-only evaluation quantities,
measured steering references, manifests, and optional listening WAVs. `run`
fits every requested K/family/condition combination. Use `--overwrite` only
when intentionally replacing an earlier preparation or completed run;
regenerating prepared data invalidates fits from the earlier preparation.

The run publishes results incrementally. After each completed
`(K, model, condition)` fit, it atomically refreshes:

- `performance_no_driving_noise.png`
- `performance_car_noise.png`

Each is exactly a two-column figure: held-out within-family log-likelihood
change versus K on the left, and held-out NMI versus K on the right. When the
K=3 fit for a family is available, the corresponding fitted-component atlas is
also refreshed:

- `components_no_driving_noise_K3.png`
- `components_car_noise_K3.png`

Atomic replacement means an existing PNG remains readable while the next
version is being rendered. Partial figures contain only fits that have actually
finished; missing points are not fabricated.

## Experimental order and fitting interface

The explicit work order is:

```text
model order K
  model family
    data condition: no_driving_noise, car_noise
      frequency
        initialization restart
```

Keeping condition inside family makes each clean/noisy pair available before
advancing to the next family. The same deterministic initialization seeds are
used for the two conditions, while restart selection uses training likelihood
only.

For every selected frequency, observations are complex four-microphone vectors:

```python
x_train.shape == (n_train_frames, 4)
x_test.shape == (n_test_frames, 4)

model = fit_model(
    x_train=x_train,
    model_name=family,
    n_components=K,
    frequency_hz=frequency_hz,
    seed=seed,
)
train_posterior = model.posterior(x_train)       # (K, n_train_frames)
test_posterior = model.posterior(x_test)         # (K, n_test_frames)
test_log_density = model.score_samples(x_test)   # (n_test_frames,)
```

These independent mixtures initialize the shared labels; they are not the final
models. Their component rows are matched from training posteriors. A joint
training posterior is then formed by multiplying component-conditional evidence
over frequencies. Each training frame receives one hard shared label, and each
frequency is refit as K separate one-component densities on those same labelled
subsets. Classification and refitting alternate for at most three iterations
(default), with a small minimum cluster size to avoid undefined covariance
fits. No source labels or held-out observations enter this procedure.

At test time, the refitted models are frozen. Their conditional evidence is
multiplied across frequency to give one K-vector posterior and one normalized
joint log density per time frame. `score_samples` from an individual frequency
is a normalized mixture log density, including mixing weights and family
normalizers; the runner removes the frequency-specific mixing weight when it
constructs the shared product mixture, whose weight occurs only once.

All families receive the same raw complex vectors. The adapter applies the
family-specific representation consistently during fitting, posterior
prediction, likelihood calculation, parameter export, and steering-template
readout:

| Family | Observation used by its density |
|---|---|
| Complex Watson | Unit-norm complex vector; common complex phase is a nuisance |
| Complex Bingham | Unit-norm complex vector |
| Complex ACG | Unit-norm complex vector |
| Uniform-marginal VMVM | All four microphone phases in radians, with the common phase marginalized by PCMM's oscillatory model |
| Complex Gaussian | Raw four-channel complex vector, with a proper zero-mean covariance mixture |

The uniform-marginal von-Mises/von-Mises circular model replaces the former
wrapped-normal quotient baseline. It is the existing PCMM VMVM implementation
with `oscillatory_data=True`; it is initialized and evaluated in that same mode.
It receives all four absolute phase coordinates because the model itself
marginalizes the common phase. Precomputing reference-channel phase differences
would instead change its sample space and density.

The Gaussian baseline is intentionally not given a time-varying source-power
model. Noise, reverberation, overlap, and speech amplitude can nevertheless
lead any family to use more than one component for a physical seat. K=3 is a
scientifically interesting reference, not an expected or forced optimum.

## Paired data construction

The default geometry and schedule are:

| Setting | Default |
|---|---|
| Distinct physical speakers | 3 |
| Occupied measured mouth positions | **1, 2, 3** |
| Display/candidate mouth positions | **1, 2, 3, 5**; seat 5 is empty |
| Selected measured microphones | Native one-based IDs **1, 4, 5, 7** |
| Source orientation | 0-degree measured mouth response |
| Training / held-out duration | 45 / 30 seconds |
| Time-frequency transform | 8 kHz; 1024-sample Hann window; 256-sample hop |
| Selected frequencies | 6 nearest FFT bins from 250 to 3500 Hz |
| Exactly two active speakers | target 8%; hard maximum 10% of final STFT frames |
| Exactly three active speakers | target 2%; hard maximum 5% of final STFT frames |
| Fitted orders | K = 1, 2, 3, 4, 5 |
| Initializations | 3 per family/K/condition/frequency |
| Driving-noise SNR | 10 dB, calibrated on training and frozen for test |

The scheduler constructs a continuous conversation: every final STFT frame has
at least one active speaker, including in the no-driving-noise condition. The
overlap categories mean **exactly** two and **exactly** three active speakers,
not “at least” those counts. The default target tolerance is 1.5 percentage
points, but the hard 10% and 5% caps are always checked. Source airtimes are
also constrained to be approximately balanced.

Only MiniLibriMix's separate `s1`/`s2` source files are used; packaged
two-speaker mixtures are not treated as clean sources. Three different speaker
IDs are chosen, and original utterances are split between training and held-out
data so an utterance cannot occur in both. This is a custom utterance-disjoint
split, not the official MiniLibriMix benchmark split. The manifest records the
speaker IDs, utterance IDs, schedule, random seed, and overlap diagnostics.

Each source is convolved with the corresponding measured multichannel mouth
impulse response. Microphones do not coincide with seats: IDs 1, 4, 5, and 7
are measured array locations above the cabin, while 1, 2, 3, and 5 identify
mouth positions. Keeping those coordinate systems distinct is intentional.

Speech clips are normalized before rendering. Then one scalar gain per source
is estimated from that source's isolated **training** image so received power
over the selected four-microphone array is approximately equal across occupied
seats. Those gains are frozen for held-out data and shared by both noise
conditions. This prevents seat/IR gain from making one talker systematically
easier without using held-out information.

For `car_noise`, measured multichannel McVAMPIRE driving recordings are selected
and resampled jointly so their spatial coherence is retained. Training and test
use disjoint files or segments. A noise gain is calibrated from training audio
at the configured SNR and then frozen for held-out audio. The clean and noisy
conditions retain identical speech samples and source-reference energies.

## No gating and the meaning of an observation

Every STFT time frame is used for fitting in both conditions. Every held-out
frame is also scored. There is no amplitude gate and no deletion of low-energy
training observations. Because the revised conversation has no silent frames,
the “test only during actual speech” rule is an all-true mask; it is still saved
and asserted so an accidental silent frame cannot silently enter evaluation.

This design deliberately exposes models to measured noise rather than training
only on oracle speech segments. It also avoids turning silence rejection into
an implicit supervised background rule. The frames overlap in time and are
correlated, so thousands of frames are not thousands of independent experimental
replications even though the mixture likelihood treats frames as observations.

## Shared labels and held-out metrics

Each frequency is initially fitted independently. Component permutations are
matched using training posterior activity only, via correlation matching and
Hungarian assignment. That match only initializes the shared train-time
partition. The final frequency-specific component densities are refit on the
same partition, so their component row k has a common learned meaning by
construction. No source labels, clean source images, schedules, candidate-seat
steering vectors, or held-out observations enter matching or refitting.

Held-out probabilities are not averaged. The final predictor is a shared-label
product mixture,

```text
p(x_t) = sum_k pi_k product_f p(x_ft | k).
```

Its joint posterior is converted to one hard component label per frame. The
target is the dominant true speaker from isolated-source power averaged over
the same frequencies and all four selected microphones. Held-out NMI compares
those two hard label sequences.

NMI is a useful component/source correspondence score, but it does not evaluate
separation of simultaneous speakers. In two- and three-speaker overlap frames,
the target still names only the dominant source. The saved overlap-stratified
counts and scores should therefore be inspected alongside the headline NMI;
single-talker NMI is the cleanest directional-recovery diagnostic.

Held-out mean log likelihood is the joint product-mixture log density averaged
over time frames; every selected frequency contributes to each frame. Euclidean,
projective, and toroidal densities live on different spaces and measures, so
their absolute likelihood values must not be compared across model families.
The left figure panel plots each family's value **relative to its own K=1
value**. This preserves meaningful within-family changes with K without
implying a common unit across representations.

If K is selected by maximum held-out likelihood, the same held-out set is being
used for model selection and reporting. It is therefore a selection score, not
an untouched final-test estimate. The true source count is marked at K=3, but
neither likelihood nor NMI is guaranteed to peak there. Components can represent
overlap, amplitude regimes, reverberant mismatch, or noise as well as seats.

## Fitted-component car atlases

The K=3 atlases are derived from the learned parameters, not from an oracle
posterior-weighted average of clean source directions. For each frequency and
component, the adapter scores four measured mouth-position steering templates
at seats 1, 2, 3, and 5 in that family's fitted representation. The now-shared
component rows are combined across frequency and normalized within each
component. The empty seat tests
whether a learned component prefers an unoccupied measured direction.

For directional families, the readout uses their fitted direction or shape
parameter. For the zero-mean Gaussian it uses the angular distribution induced
by the fitted covariance, rather than raw impulse-response amplitude. For VMVM
it evaluates the fitted circular phase interaction on each steering template.
The fitted parameters and four discrete support scores are saved with the model
result so the readout can be audited.

The shaded car contours are only a schematic interpolation between those four
measured support values. Four discrete templates cannot identify a continuous
physical mean and covariance over the cabin, and model covariance lives in
complex observation space rather than car-coordinate space. Accordingly, the
atlas is labelled “relative steering score” and explicitly warns that it is not
a fitted localization covariance or calibrated location probability. The car
outline and measured coordinates aid interpretation; they do not add spatial
evidence.

## Outputs and reproducibility

The prepared metadata records the preparation ID, complete configuration,
geometry, selected files, utterance-disjoint split, activity schedule, overlap
fractions, balancing gains, noise calibration, and train/test diagnostics.
Condition NPZ files contain raw complex arrays, common time/frequency axes,
speech/source activity, isolated source energies for evaluation, and all-true
validity masks. Reference steering data is never passed into the fitter.

The results directory contains an incrementally written `metrics.csv` and
`progress.json`, per-condition/per-family/per-K NPZ files, the four PNGs listed
above, and a final `results.json` completion marker. Per-fit files preserve the
initial frequency matching, initial and final per-frequency posteriors, shared
train/test posteriors, joint framewise likelihoods, consensus diagnostics,
learned parameter exports, component steering scores, and evaluation masks.

Run the automated checks with:

```bash
python -m unittest -v test_experiment
python -m unittest -v test_fitter_adapter
```

These tests check construction and plumbing, not whether a distribution wins or
whether K=3 is selected. See `VALIDATION.md` for the boundary between verified
software behavior and the still-pending full scientific run.

Dataset records: [MiniLibriMix](https://zenodo.org/records/3871592) and
[In-Car McVAMPIRE V1.0](https://zenodo.org/records/12806684). Neither audio
corpus is redistributed here.
