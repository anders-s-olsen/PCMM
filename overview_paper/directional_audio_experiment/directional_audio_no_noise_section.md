## Speaker diarization in a car

We next evaluated whether the fitted mixture components could recover the
physical direction of speech sources in a measured multichannel acoustic
environment. Three different MiniLibriMix speakers were placed at mouth
positions 1, 2, and 3 of the In-Car McVAMPIRE geometry. Each source signal was
convolved with the corresponding measured mouth-to-microphone impulse
responses, and microphones 1, 4, 5, and 7 were retained. Mouth position 5 was
left empty and used only as a negative-control support point when visualizing
the spatial preferences of the learned components. The observations in this
experiment contain the three reverberant speech images without an added
background recording.

Training and held-out recordings were 45 s and 30 s long, respectively, and
used disjoint source utterances. The conversation schedule was continuously
active: every analysis frame contained at least one speaker. Approximately
90% of the frames contained one active speaker, 8% contained two, and 2%
contained all three. Speaker airtime was balanced by construction. To prevent
the gain of a particular measured impulse response from making its source
systematically easier to identify, one scalar gain per source was estimated
from its isolated training image such that the received training powers were
equal over the selected microphone array. These gains were then held fixed for
the held-out recording. Consequently, the experiment tests spatial structure
rather than a fixed between-seat level difference, while retaining the natural
within-utterance variation in speech power.

The signals were sampled at 8 kHz and transformed with a 1024-sample Hann
window and a 256-sample hop. We retained six Fourier bins, centered at 250,
898, 1547, 2203, 2852, and 3500 Hz. Thus, each observation at one frequency and
time was a complex four-microphone vector. All 1403 training frames and all 934
held-out frames were used; no amplitude gate or oracle speech-selection rule
was applied. The frequency-dependent source powers shown in Fig. Xb were
computed from the isolated rendered source images for visualization and
evaluation only and were not supplied to any fitted model.

For every model family, we fitted mixture orders $K=1$ to $5$ independently
at each of the six frequencies. Three initializations were considered for each
frequency, and the initialization with the largest training likelihood was
retained. The frequency-specific component permutations were first aligned
using their training responsibilities. To give a component one shared meaning
across frequencies, we then alternated between classifying each training frame
under a shared-label product mixture and refitting the frequency-specific
component densities to the resulting common partition. At most three such
consensus updates were performed. All alignment, initialization selection,
classification, and refitting used training observations only. The fitted
models were frozen before the held-out recording was classified.

We assessed component recovery using normalized mutual information (NMI).
At each frequency, isolated source power was averaged over the four
microphones and converted to a fraction of the total isolated-source power;
the reference label was the speaker with the largest fraction after averaging
over the six frequencies. The predicted label was the maximum-posterior
component of the shared-label product mixture. NMI is invariant to a global
permutation of component indices and therefore does not require an oracle
component-to-speaker relabeling. In overlap frames it evaluates only the
dominant speaker; it is not a source-separation score and does not assess
whether every simultaneously active speaker was recovered. We show training
and held-out NMI rather than log likelihood because the model families are
defined on different sample spaces and their absolute likelihoods are not
comparable, while within-family likelihood improvements need not correspond to
recovery of the physical sources.

The directional families showed a clear maximum in source recovery at the
true number of occupied positions, $K=3$ (Fig. Xc). At this order, held-out
NMI was 0.945 for complex Bingham, 0.944 for complex ACG, 0.933 for complex
Watson, and 0.869 for the uniform-marginal von Mises circular model (UMVM).
The corresponding
training values were 0.967, 0.967, 0.960, and 0.889, respectively, showing that
the recovered structure was not confined to the training recording. The
same conclusion is visible in the spatial readout (Fig. Xd): for each of these
four models, the three displayed components had distinct maxima at occupied
positions 1, 2, and 3, while the empty position 5 received little support.
ACG, UMVM, and Watson met the consensus stopping criterion at this order. The
final Bingham update changed only 2 of 1403 training labels (0.14%), narrowly
above the configured 0.1% threshold.

The changes with $K$ are also consistent with a three-source scene. With only
two components, one component must combine at least two dominant-speaker
classes; held-out NMI therefore remained between 0.626 and 0.705 for the four
directional families. At $K=3$, their held-out NMI increased to
0.869--0.945. Increasing the order beyond the physical source count reduced
NMI: at $K=4$, it was 0.876 for ACG, 0.795 for Watson, 0.784 for Bingham, and
0.729 for UMVM. Extra components can improve a density model by representing
within-source heterogeneity, reverberant deviations, overlap frames, or
different signal regimes, but NMI penalizes the resulting subdivision of one
physical speaker into multiple component labels. Thus, model order in this
experiment is not intrinsically identical to source count; the coincidence at
$K=3$ is an empirical recovery result rather than a fitted constraint.

The complex Gaussian behaved differently. Its held-out NMI was 0.089 at
$K=2$, 0.368 at $K=3$, and reached only 0.564 at $K=4$. At $K=3$, the
spatial readout contained a clear component for position 1 and two components
whose angular preferences were concentrated near position 3. The component
assigned to the position-2 display column placed only 0.103 of its pooled
four-seat score on position 2 and 0.832 on position 3. Accordingly, position 2
did not receive a distinct Gaussian component.

This failure is explained by the additional radial information retained by the
Gaussian observation model. The directional models operate on normalized
vectors or phases and are therefore insensitive to the large frame-to-frame
variation in speech magnitude. In contrast, the Gaussian was fitted to the raw
complex STFT vectors with one fixed rank-one-plus-isotropic covariance per
component and no frame-specific source-power variable. Speech magnitudes are
strongly non-Gaussian and vary substantially within a speaker. The fitted
Gaussian partition consequently separated power regimes as well as spatial
directions: the median total powers of its three $K=3$ training clusters were
approximately -9.2, 0.8, and 7.8 dB relative to the median power over all
training frames. This approximately 17 dB separation used component capacity
that would otherwise have distinguished the three seats. The contrast with
ACG is particularly informative because ACG retains the angular information
associated with a complex covariance while removing observation norm; its
held-out NMI was 0.944 on the same data.

The Gaussian result also carries an optimization caveat. Its fraction of
training labels changed by 13.6%, 7.8%, and 2.7% over the three allowed
consensus updates, so it had not reached the configured 0.1% convergence
criterion. This incomplete stabilization may affect the exact Gaussian NMI and
the allocation of its components. It does not, however, remove the observed
power stratification or the central model mismatch: a fixed-scale Gaussian
must explain both direction and the broad radial distribution, whereas the
directional models condition away that radial nuisance. A compound-Gaussian
model with a frame-varying scale, or an explicit source-power model, would be a
more appropriate Euclidean baseline if magnitude is to be retained.

Taken together, the results show that the measured multichannel phase and
relative-amplitude structure is sufficient to identify the three reverberant
source positions in held-out speech. Recovery is strongest when the component
family treats overall STFT magnitude as a nuisance and when the number of
components matches the number of occupied positions. Because this experiment
contains one fixed geometry and one constructed train--held-out scene, the
reported NMI values are descriptive comparisons for this scene rather than
population-level estimates; the STFT frames are overlapping and should not be
treated as independent replicates.

### Figure legend

**Figure X. Speaker diarization in measured car acoustics.** **a**, The
five-seat saloon layout cropped from the measured In-Car McVAMPIRE geometry.
Colored positions 1, 2, and 3 contain independent speakers; position 5 is
empty. The four selected microphones are labeled m1--m4 for display and
correspond to native measured channels 1, 4, 5, and 7. **b**, Isolated
per-source power in the first 18 s of the training and held-out recordings.
The six colors give the separate contributions at 250, 898, 1547, 2203, 2852,
and 3500 Hz; frequencies are not grouped. Heights are normalized by a
source-specific 95th percentile estimated from the training recording and
clipped at one. The complete 45 s and 30 s recordings were used for fitting
and evaluation. **c**, NMI between the dominant source and the shared
component label in training and held-out frames as a function of mixture
order. The dashed line marks the three occupied positions. **d**, Spatial
scores for the $K=3$ components.
Components are ordered for display by their association with occupied
positions 1, 2, and 3. Color denotes the component's relative score at the
four measured candidate positions after pooling the six frequencies;
contours are inverse-distance interpolations included only to connect these
support points within the five-seat saloon. They are not location
probabilities or fitted spatial covariance contours. Colored outlines identify
occupied positions, the dashed outline identifies empty position 5, and
triangles identify the selected microphones. The display ordering does not
affect NMI. UMVM denotes the uniform-marginal von Mises circular model.
