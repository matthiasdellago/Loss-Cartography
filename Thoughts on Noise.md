
##### The loss landscape is noisy on the scale of plausible learning rates.
The experiments show that when we use standard precision in models (float32), and try to approximate the curvature via finite differences
$$
f''(x) \approx \frac{f(x) - 2f(x+h) + f(x+2h)}{h^2}
$$
The finite difference $f(x) - 2f(x+h) + f(x+2h)$ contains inherent noise on the order of $10^{-8}$, which becomes relevant for $h$ on the order of $10^{-3}$, which is well within the range of common learning rates, maybe even on the large side.
(For torch.float64 these exponents are approxemately doubled ($10^{-15}$ and $10^{-6}$, and the noise threshold is therefore much lower.)

The fact that it is shaped like (Gaussian) 'noise' and **not** quantisation (ie. rounding the loss), is probably due to accumulation of independent errors all along the forward calculation (PROVE). Corrolary: The wider and deeper the network the larger the noise! Probably $O(\sqrt{n})$, central-limit-theorem style.
Likely, the gradient is even noisier, since it additionally accumulates error on the backward. The earlier the layer, the more noise it recieves in its gradient (PROVE and EXPERIMENT). 

To me, this suggests that below $10^{-3}$ the jacobian is no longer good at predicting the loss, not because of batch noise (we evaluate on the complete dataset) but purely numerical noise. Since much of deep learning happens within this reginme this seems worth investigating.
Especially with bfloat16 and other even lower precision types this will be highly relevant, especially if the network width and depth increases the noise!
Maybe the noise is a feature rather than a bug, creating langevin dynamics in otherwise pure SGD (EXPERIMENT, LITERATURE REVIEW).

#### Researche paths from here 
1. Given precision, width and depth, what noise level can I expect in my gradient? 
	1. And what effect does it have empirically and theoretically? ie: What is SGD in the limit of low precision? learning rate $\approx$ noise level
2. Now that we understand the noise portion of the roughness/scale graph: excellent!
	1. Understand the quadratic term -> Linear Regression
	2. Account for both, and then investigate original objective: Scale of structure
3. 


