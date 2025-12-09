from __future__ import annotations

import numpy as np

def compute_expected_profit():
    dtype = np.longdouble
    ys = np.arange(0, 10, dtype=int)
    expectations = np.zeros_like(ys, dtype=dtype)
    for y in range(len(expectations)):
        init = y+1
        for payment in range(init, 11):
            if payment == init:
                expectations[y] += payment * ((-2 * payment * payment + 22 * payment -1)/100)
            else:
                expectations[y] += payment * ((-2*payment+21)/100)
    return ys, expectations

def main():
    prec = 12
    fmt = f"{{:.{prec}f}}"
    ys, exps = compute_expected_profit()
    for y, val in zip(ys, exps):
        print(f"y={y+1:2d} -> E[X|Y={y+1}] = {fmt.format(val)}")
    best_idx = int(np.argmax(exps))
    best_val = exps[best_idx]
    print(f"Best y = {best_idx+1}, E[X|Y={best_idx+1}] = {fmt.format(best_val)}\n")

if __name__ == "__main__":
	main()