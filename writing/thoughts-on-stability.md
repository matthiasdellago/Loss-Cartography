The Gradients must flow!
thoughts on stability in deep neural networks

Basically we want:

$$
\frac{\partial \text{loss}}{\partial \text{params}} = 0, \text{ i.e., optimal}
$$
$$
\frac{\partial \text{output}}{\partial \text{input}} \approx 1, \text{ i.e., stable}
$$
Sensitivity in unputs vs sensitivity in parameters.


Other names for stability issues:
- Exploding/vanishing gradient problem
- Mode collapse in genAI
- dying ReLU problem

Most fixes things are about stability:
- weight decay: better matrix conditioning [Tikhonov]
	- Magnitude often unaffected, as per some paper
	- Experimentally confirm effect on singular values
- initialisation: init to orthogonal matrices -> deeep network [Xiao,2018]
- non-linieraities
	- What does stability mean for non-linearities?
	- Additional consideration: Convex and monotonic non-liniearities preserve convexity, if input is convex. Very good for optimisation.
		- Gelu/swish dont do this?
- residual connections: conditioning
- layer norm
	- actually nowadays RMSnorm is used, which literally just norms the vector to unit euclidian length
	- 



Bonus: 
- What does this mean in the loss landscape?
- What does adam do about this? Question: Does adam average more over trajectory or over different batches? which variance dominates? maybe the optimal learning rate is in either one regime or exactly at the tradeoff?