"""
NumPy-accelerated computation of E[X | Y=y] for y=0..10.

This implementation uses NumPy vectorized operations and attempts to use
extended precision via ``np.longdouble`` when available. The conditional
probability p(X|Z,Y) follows the mapping described by the user; for
each (z,y) the mass over X in [10-y, 10-z] is assigned using cumulative
tails at the endpoints and individual base probabilities in the interior.

Run as a script to print E[X|Y=y] for y=0..10 and the maximizing y.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def compute_expectations(pz=None, dtype=None) -> Tuple[np.ndarray, np.ndarray]:
	"""Compute E[X|Y=y] for y=0..10 using vectorized NumPy operations.

	Args:
		pz: optional sequence of length 11 giving P(Z=0),...,P(Z=10). If None,
			defaults to the legacy uniform weight 1/11 for each z (this matches
			previous behavior where a factor 1/11 was applied after summation).
		dtype: numeric dtype to use (e.g., np.longdouble). If None, function
			will try to use np.longdouble and fall back to np.float64.

	Returns:
		ys: array of y values (0..10)
		expectations: array of E[X|Y=y] with dtype provided
	"""
	# Choose dtype: try longdouble for higher precision, else default float64
	if dtype is None:
		try:
			dtype = np.longdouble
		except Exception:
			dtype = np.float64

	# Base counts provided (denominator 1024)
	base_counts = np.array([1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1], dtype=dtype)
	denom = dtype(1024)
	baseP = base_counts / denom  # normalized base probabilities

	ys = np.arange(0, 11, dtype=int)
	expectations = np.zeros_like(ys, dtype=dtype)

	# pz: probability mass function for Z. If not provided, default to legacy
	# uniform weight 1/11 for each z to match prior implementation behavior.
	if pz is None:
		pz_arr = np.full(11, dtype(1) / dtype(11), dtype=dtype)
	else:
		pz_arr = np.array(pz, dtype=dtype)
		if pz_arr.shape[0] != 11:
			raise ValueError("pz must be length 11 (probabilities for Z=0..10)")

	# Precompute prefix/suffix sums for speed
	prefix = np.cumsum(baseP)  # prefix[i] = sum_{0..i} baseP
	suffix = np.cumsum(baseP[::-1])[::-1]  # suffix[i] = sum_{i..10} baseP

	for y in range(0, 11):
		total = dtype(0)
		x_low = 10 - y
		# z runs from 0..y
		zs = np.arange(0, y + 1, dtype=int)
		for z in zs:
			x_high = 10 - z
			# vector of x values in the allowed range [x_low, x_high]
			x_vals = np.arange(x_low, x_high + 1, dtype=dtype)
			# map to index k = 10 - x
			ks = (10 - x_vals).astype(int)

			# initialize p-array
			p = np.zeros_like(x_vals, dtype=dtype)

			if z == y:
				# single value -> probability 1
				p[:] = dtype(1)
			else:
				# assign endpoint masses using prefix/suffix
				# where k == z: sum_{0..z} baseP -> prefix[z]
				mask_z = (ks == z)
				if mask_z.any():
					p[mask_z] = prefix[z]

				# where k == y: sum_{y..10} baseP -> suffix[y]
				mask_y = (ks == y)
				if mask_y.any():
					p[mask_y] = suffix[y]

				# interior points: z < k < y -> baseP[k]
				mask_interior = (ks > z) & (ks < y)
				if mask_interior.any():
					p[mask_interior] = baseP[ks[mask_interior]]

			# accumulate x * p weighted by P(Z=z)
			total += pz_arr[z] * np.sum(x_vals * p)

		expectations[y] = total

	return ys, expectations


def main() -> None:
	# Attempt to use longdouble for improved precision
	try:
		dtype = np.longdouble
	except Exception:
		dtype = np.float64

	# Helper: build a truncated discrete Gaussian pmf over {0..10}
	def truncated_discrete_gaussian_pz(sigma: float, mu: float = 5.0) -> np.ndarray:
		"""Return an array pz of length 11 with P(Z=k) proportional to
		exp(- (k-mu)^2 / (2 sigma^2)) for k=0..10, normalized to sum to 1.
		"""
		ks = np.arange(0, 11, dtype=dtype)
		# avoid divide-by-zero when sigma is extremely small
		if sigma <= 0:
			raise ValueError("sigma must be positive")
		exps = np.exp(-((ks - dtype(mu)) ** 2) / (2 * (dtype(sigma) ** 2)))
		pz = exps / np.sum(exps)
		return pz

	# Print with higher precision
	prec = 12
	fmt = f"{{:.{prec}f}}"

	# Baseline: legacy uniform pz = 1/11 for all z
	print("--- Baseline: uniform pz (legacy 1/11) ---")
	legacy_pz = np.full(11, dtype(1) / dtype(11), dtype=dtype)
	ys, exps = compute_expectations(pz=legacy_pz, dtype=dtype)
	for y, val in zip(ys, exps):
		print(f"y={y:2d} -> E[X|Y={y}] = {fmt.format(val)}")
	best_idx = int(np.argmax(exps))
	best_val = exps[best_idx]
	print(f"Best y = {best_idx}, E[X|Y={best_idx}] = {fmt.format(best_val)}\n")

	# Now evaluate for several sigma values using the truncated discrete Gaussian pmf
	for sigma in (1.0, 2.0, 5.0):
		print(f"--- Truncated discrete Gaussian pz with sigma={sigma} ---")
		pz = truncated_discrete_gaussian_pz(sigma=sigma, mu=5.0)
		# show the pmf values for Z=0..10
		print("pz:", np.array2string(pz, formatter={"float_kind": lambda x: f"{x:.6f}"}))
		ys, exps = compute_expectations(pz=pz, dtype=dtype)
		for y, val in zip(ys, exps):
			print(f"y={y:2d} -> E[X|Y={y}] = {fmt.format(val)}")
		best_idx = int(np.argmax(exps))
		best_val = exps[best_idx]
		print(f"Best y = {best_idx}, E[X|Y={best_idx}] = {fmt.format(best_val)}\n")


if __name__ == "__main__":
	main()


