# Compatibility Note for Old Checkpoints

During the ICLR rebuttal stage, we made a small implementation change to RoPE. 
This change does not affect the core logic of the method, 
but it may cause slight differences when earlier checkpoints (`Old-XXX.pth`) are loaded with the current codebase.

For example, when evaluating `Old-SD1-10x05.pth` on the FJSP datasets:

```python
SD1-10x05  12.25% (original version) -> 12.15% (rebuttal version)
Brandimarte 9.08% (original version) -> 9.59% (rebuttal version)
Hurink(vdata) 3.48% (original version) -> 3.36% (rebuttal version)
```

To exactly reproduce the results reported in the main paper, we also provide the original model file. In particular, please:
If you want to completely replicate the results of the main text, we also provide the original model file. Specifically, you need to:
- copy the `SchedulingModel_old.py` file to the corresponding folder;
- modify the imported model and method in `SchedulingRunner.py` and `SchedulingEvaluator.py` with `SchedulingModel_old`.
