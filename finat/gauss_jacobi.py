"""Basic tools for getting Gauss points and quadrature rules."""
__copyright__ = "Copyright (C) 2014 Robert C. Kirby"
__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import math
from math import factorial
import numpy


def compute_gauss_jacobi_points(a, b, m):
    """Computes the m roots of P_{m}^{a,b} on [-1,1] by Newton's method.
    The initial guesses are the Chebyshev points.  Algorithm
    implemented in Python from the pseudocode given by Karniadakis and
    Sherwin"""
    x = []
    eps = 1.e-8
    max_iter = 100
    for k in range(0, m):
        r = -math.cos((2.0 * k + 1.0) * math.pi / (2.0 * m))
        if k > 0:
            r = 0.5 * (r + x[k - 1])
        j = 0
        delta = 2 * eps
        while j < max_iter:
            s = 0
            for i in range(0, k):
                s = s + 1.0 / (r - x[i])
            f = eval_jacobi(a, b, m, r)
            fp = eval_jacobi_deriv(a, b, m, r)
            delta = f / (fp - f * s)

            r = r - delta
            if math.fabs(delta) < eps:
                break
            else:
                j = j + 1

        x.append(r)
    return x


def gauss_jacobi_rule(a, b, m):
    xs = compute_gauss_jacobi_points(a, b, m)

    a1 = math.pow(2, a + b + 1)
    a2 = math.gamma(a + m + 1)
    a3 = math.gamma(b + m + 1)
    a4 = math.gamma(a + b + m + 1)
    a5 = factorial(m)
    a6 = a1 * a2 * a3 / a4 / a5

    ws = [a6 / (1.0 - x ** 2.0) / eval_jacobi_deriv(a, b, m, x) ** 2.0
          for x in xs]

    return numpy.array(xs), numpy.array(ws)


def eval_jacobi(a, b, n, x):
    """Evaluates the nth jacobi polynomial with weight parameters a,b at a
    point x. Recurrence relations implemented from the pseudocode
    given in Karniadakis and Sherwin, Appendix B"""

    if 0 == n:
        return 1.0
    elif 1 == n:
        return 0.5 * (a - b + (a + b + 2.0) * x)
    else:  # 2 <= n
        apb = a + b
        pn2 = 1.0
        pn1 = 0.5 * (a - b + (apb + 2.0) * x)
        p = 0
        for k in range(2, n + 1):
            a1 = 2.0 * k * (k + apb) * (2.0 * k + apb - 2.0)
            a2 = (2.0 * k + apb - 1.0) * (a * a - b * b)
            a3 = (2.0 * k + apb - 2.0)  \
                * (2.0 * k + apb - 1.0) \
                * (2.0 * k + apb)
            a4 = 2.0 * (k + a - 1.0) * (k + b - 1.0) \
                * (2.0 * k + apb)
            a2 = a2 / a1
            a3 = a3 / a1
            a4 = a4 / a1
            p = (a2 + a3 * x) * pn1 - a4 * pn2
            pn2 = pn1
            pn1 = p
        return p


def eval_jacobi_deriv(a, b, n, x):
    """Evaluates the first derivative of P_{n}^{a,b} at a point x."""
    if n == 0:
        return 0.0
    else:
        return 0.5 * (a + b + n + 1) * eval_jacobi(a + 1, b + 1, n - 1, x)
