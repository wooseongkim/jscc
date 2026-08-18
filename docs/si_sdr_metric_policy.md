# SI-SDR evaluation policy

Historical J1–J4/J5/R4 artifacts retain their original full-crop SI-SDR values.
They are not rewritten.

New waveform experiments must use:

```python
from speech_jscc.evaluation.si_sdr_alignment import aligned_waveform_metrics
metrics = aligned_waveform_metrics(reference, estimate, sample_rate, max_lag_ms=5.0)
```

The aligned diagnostic searches positive normalized cross-correlation in a
±5 ms window, scores the overlapping waveforms, and records the selected lag.
Clean codec and JSCC outputs must use the same call and lag window. The gate
must remain based on the aligned clean-codec-relative values for that new run;
historical gates are not retroactively redefined.
