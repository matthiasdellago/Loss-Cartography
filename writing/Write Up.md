### Method

##### Theory
Let's find *characteristic scales* in the loss ladscapes.
Our starting point is the finite difference approximation of curvature for a given 'scale' $h$:
$$
f''(x) \approx \frac{f(x) - 2f(x+h) + f(x+2h)}{h^2}
$$
From inspection and dimensional analysis it is obvious that this is not scale independent - A curve scaled up by a factor of two, will have half the curvature. For a fair comparison of scales we will multiply by $h$:
$$
Roughness(x,h) := \frac{f(x) - 2f(x+h) + f(x+2h)}{h}\quad(\approx f''(x) \cdot h)
$$

NOTE: Should we instead define it as the deviation from the tangent space of the loss? How much does the $f(x+h)$ deviate from $f(x)+h \cdot f'(x)$. More elegant?
##### Implementation
1. Pick a point in parameter space.
2. Pick several (random)directions leading away from the point. (including grad ascent/decent, and towards/away from $\vec{0}$)
3. Pick a minimum distance.
4. Sample the loss in each direction in doubling distances.
5. Plug into $Roughness(x,h)$
### Results

#### SimpleMLP on MNIST
![[abs(Curvature) of Simple MLP on MNIST.png]]
![[abs(Grit) of Simple MLP on MNIST.png]]
![[abs(Finite Difference) of Simple MLP on MNIST.png]]

#### Linear Regression (Pytorch)
![[_Curvature_ of Linear Regression.png]]
![[abs(Grit) of Linear Regression.png]]
![[_Finite Difference_ of Linear Regression.png]]
Looks almost the same! Why?
#### Linear Regression (Analytical)
![[quadratic+noise.png]]
They're both just a quadratic potential with noise.
BIG QUESTION 1: Noise is numeric. But why quadratic?
	- Investigate different loss criteria.
BIG QUESTION 2: Try to find roughness we were originally looking for? Or pivot to understanding the noise and its implications?
### Conclusion
##### The loss landscape is noisy on the scale of plausible learning rates.
The experiments show that when we use standard precision in models (float32), and try to approximate the curvature via finite differences
$$
f''(x) \approx \frac{f(x) - 2f(x+h) + f(x+2h)}{h^2}
$$
The finite difference $f(x) - 2f(x+h) + f(x+2h)$ contains inherent noise on the order of $10^{-8}$, which becomes relevant for $h$ on the order of $10^{-3}$. (For torch.float64 these exponents are approxemately doubled ($10^{-15}$ and $10^{-6}$, and the noise threshold is therefore much lower.) $10^{-3}$ is a common if small learning rate when using SGD, although I don't know how large the actual steps are.

The fact that it is shaped like (Gaussian) 'noise' and **not** quantisation (ie. rounding the loss), is probably due to accumulation of independent errors all along the forward calculation (PROVE). Corrolary: The wider and deeper the network the larger the noise! Probably $O(\sqrt{n})$, central-limit-theorem style. Each layer contributes some numerical error, and magnifies the error in its input by its conditioning number! (calculate conditioning number for matrices and nonlinearities).
This loss in precision is another way to view the exploding/vanishing gradient problem -- It's just float overflow/underflow!
The gradient is even noisier, since it additionally accumulates error on the backward. The earlier the layer, the more noise it recieves in its gradient (PROVE and EXPERIMENT). 

To me, this suggests that below $10^{-3}$ the jacobian is no longer good at predicting the loss, not because of batch noise (we evaluate on the complete dataset) but purely numerical noise. Since much of deep learning happens within this reginme this seems worth investigating.
Especially with bfloat16 and other even lower precision types this will be highly relevant, especially if the network width and depth increases the noise!
Maybe the noise is a feature rather than a bug, creating langevin dynamics in otherwise pure SGD (EXPERIMENT, LITERATURE REVIEW).

Misc Insights
- CONJECTURE: weight decay is just for maintaining a small gap between the largest and smallest singular value of a matrix -> good condition number!!
- We should be able decrease precision until numerical noise is just below the order of batch noise!
	- Momentum/adam helps with averaging out noise (langevin dynamics)
- What does layernorm/batchnorm do to the condition number?

### Future Research
1. Given precision, width and depth, what noise level can I expect in my gradient? 
	1. And what effect does it have empirically and theoretically? ie: What is SGD in the limit of low precision? learning rate $\approx$ noise level
2. Now that we understand the noise portion of the roughness/scale graph: excellent!
	1. Understand the quadratic term -> Linear Regression
	2. Account for both, and then investigate original objective: Scale of structure



