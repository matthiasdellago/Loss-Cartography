1. Lower precision increases roughness at smaller scales.
	1. Hypothesis: CONFIRMED ![[it's_working.png]]This V-Shape comes from a polynomial (harmonic) potential, with noise caused by floating point precision. Unit of least precision of the loss / dist.
2. Replicate Tensor Programs V: wider Networks cause more roughness?
3. Compare trained params, vs random params.
	- Visualising paper: Look in direction of another minima, and see whats inbetween.
4. Replicate: Residual connections reduce roughness.
5. How noisy is the gradient? Compare float32 and float64.

