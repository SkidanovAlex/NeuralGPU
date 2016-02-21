# NeuralGPU

This is an implementation of a NeuralGPU based on [this paper](http://arxiv.org/abs/1511.08228)

It slightly diverges from the paper in some small details (doesn't have gradient noise, but does have L2-regularization), and uses slightly different curriculum schedule that the original NeuralGPU.

It does converge very well for simple tasks such as reversing the sequence. It converges up to 18 digits for multiplication. Presumably with some minor tuning it will converge for larger numbers, but I don't have enough resources to debug it (since converging to 18 digits takes approximately 10 hours on a g2 instance, and AWS wouldn't let me have more than 5 of those).

Interestingly, it has harder time learning addition than multiplication. Only once in five runs it learns beyond 10 digits, while multiplication almost consistently reaches 16.

Usage
-----

```
python neuralgpu.py add
python neuralgpu.py reverse
python neuralgpu.py mul
```

It will report progress every iteration.

Curriculum schedule
-------------------
The curriculum is

```
1. Current length
2. Random up to current length favoring larger numbers
3. Current length
4. Random up to current length
5. Current length
6. Random up to 20
repeat
```
The length is increased as soon as at least one iteration for the current length reaches 0.998 accuracy in terms of number of bits properly predicted in a batch.

