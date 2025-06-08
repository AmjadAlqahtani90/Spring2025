from gmpy2 import mpz, isqrt, invert, digits, powmod

def F(n):

    N = mpz(n) # convert n to mpz  precision interger
    A = isqrt(N) + 1 
    x_squared = A**2 - N # x^2 = A^2 - N
    x = isqrt(x_squared)
    # make sure that they are maching
    if x * x != x_squared:
        raise ValueError("x^2 is not a perfect square — Fermat's method failed.")

    p = A - x
    q = A + x

    assert p * q == N, "Factorization failed: p * q != N" # checking assert

    return p, q, x

n = mpz('179769313486231590772930519078902473361797697894230657273430081157732675805505620686985379449212982959585501387537164015710139858647833778606925583497541085196591615128057575940752635007475935288710823649949940771895617054361149474865046711015101563940680527540071584560878577663743040086340742855278549092581')
p, q, x = F(n)
print("p =", p)
print("q =", q)
print("x =", x)

print("verify the N = p*q", n == p*q)
