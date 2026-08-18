# R4 Global Balanced Triplets Design

R4 extends the evaluation-only physical OFDM engine to 512 FFT bins, 360
active subcarriers, 36-sample CP, and 9,720 candidate data RE. Six channel
coefficients retain the R3 physical delays in seconds and are placed at R4
sample delays `[0,2,4,6,8,10]`.

The allocator treats all candidate RE as one pool, maintains deterministic
per-subcarrier occupancy, and constructs 1,920 triplets spanning separated
active-subcarrier regions. Triplets are balanced before stratified provisional
layer assignment. Weak triplets receive bounded inverse-square-root total
power; every branch retains at least 0.15 of its triplet fraction. Total
expected packet energy remains exactly 5,760.

Current receiver CSI and the validated raw-observation coherent MRC equation
remain unchanged. No training, clamp, erasure, coding, jammer, or production
path change is permitted.
