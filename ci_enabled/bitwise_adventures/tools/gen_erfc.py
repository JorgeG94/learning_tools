"""Coefficient generator for erfc_reprod: everything in stdlib Decimal at 80+
digits, no external deps.  Emits Fortran parameter lists and verifies each
band's fit against the Decimal reference before printing."""
from decimal import Decimal as D, getcontext

getcontext().prec = 100
PI = D('3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651')
SQRTPI = PI.sqrt()
TWO_OVER_SQRTPI = 2 / SQRTPI

def dcos(x):
    # Taylor cosine, |x| <= pi
    s, term, n = D(1), D(1), 0
    while abs(term) > D('1e-95'):
        n += 2
        term *= -x*x / (n*(n-1))
        s += term
    return s

def derfc(x):
    # series erfc(x) = 1 - 2/sqrt(pi) sum (-1)^n x^(2n+1) / (n! (2n+1)), x >= 0
    s, term, n = D(0), x, 0
    while True:
        s += term / (2*n + 1)
        n += 1
        term *= -x*x / n
        if abs(term) < D('1e-90') and n > 5: break
        if n > 600: raise RuntimeError('no convergence')
    return 1 - TWO_OVER_SQRTPI * s

def derfcx(x):
    return derfc(x) * (x*x).exp()

def cheb_fit(f, a, b, N=72, keep=40):
    a, b = D(a), D(b)
    nodes, fv = [], []
    for k in range(N):
        th = PI * (2*D(k) + 1) / (2*N)
        xk = (a+b)/2 + (b-a)/2 * dcos(th)
        nodes.append(th); fv.append(f(xk))
    coefs = []
    for j in range(keep):
        s = D(0)
        for k in range(N):
            s += fv[k] * dcos(j*nodes[k])
        c = 2*s/N
        coefs.append(c/2 if j == 0 else c)
    # trim to target decay
    while abs(coefs[-1]) < D('1e-21'): coefs.pop()
    return coefs

def clenshaw(coefs, a, b, x):
    a, b = D(a), D(b)
    t = (2*x - (a+b)) / (b-a)
    b1 = b2 = D(0)
    for c in reversed(coefs[1:]):
        b1, b2 = 2*t*b1 - b2 + c, b1
    return t*b1 - b2 + coefs[0]

def verify(coefs, f, a, b, n=151):
    worst = D(0)
    for i in range(n):
        x = D(a) + (D(b)-D(a))*i/(n-1)
        approx, exact = clenshaw(coefs, a, b, x), f(x)
        rel = abs((approx-exact)/exact)
        worst = max(worst, rel)
    return worst

def emit(name, vals):
    print(f"  real(wp), parameter :: {name}(0:{len(vals)-1}) = [ &")
    for j in range(0, len(vals), 3):
        chunk = ", ".join(f"{float(v)!r}_wp" for v in vals[j:j+3])
        tail = ", &" if j+3 < len(vals) else " ]"
        print("    " + chunk + tail)

# --- band A: erf Maclaurin, |x| <= 0.46875: erf = x * P(x^2)
mac = []
fact = D(1)
for n in range(0, 16):
    if n: fact *= n
    mac.append(TWO_OVER_SQRTPI * (-1)**n / (fact * (2*n+1)))
emit('erf_mac', mac)

# --- band B: erfcx on [0.46875, 2]
cB = cheb_fit(derfcx, '0.46875', '2')
print(f"! band B: {len(cB)} coefs, max rel fit err = {float(verify(cB, derfcx, '0.46875', '2')):.3e}")
emit('cxB', cB)

# --- band C: erfcx on [2, 7]
cC = cheb_fit(derfcx, '2', '7')
print(f"! band C: {len(cC)} coefs, max rel fit err = {float(verify(cC, derfcx, '2', '7')):.3e}")
emit('cxC', cC)

# --- band D: asymptotic a_n = (-1)^n (2n-1)!! / 2^n, in s = 1/x^2
asy, df = [D(1)], D(1)
for n in range(1, 21):
    df *= (2*n - 1)
    asy.append((-1)**n * df / D(2)**n)
emit('asyD', asy)

# constants
print(f"  real(wp), parameter :: one_over_sqrtpi = {float(1/SQRTPI)!r}_wp")
print(f"! sanity: erfc(1)  = {float(derfc(D(1))):.17e}")
print(f"! sanity: erfc(5)  = {float(derfc(D(5))):.17e}")
print(f"! sanity: erfcx(7) = {float(derfcx(D(7))):.17e}")
