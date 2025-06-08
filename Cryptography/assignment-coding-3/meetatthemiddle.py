
import gmpy2

gmpy2.get_context().precision=1530

def discrete_log(p, g, h):

    B = pow(2, 20)

    # step 1
    hashtable_X1 = {}

    for x1 in range(B):
        g_x1 = gmpy2.powmod(g, x1, p)
        inv_g_x1 = gmpy2.invert(g_x1, p)
        x = (h * inv_g_x1) % p
        hashtable_X1[x] = x1
        if x1 % 100000 == 0:
            print(f"Step 1 progress: x1 = {x1}/{B}")
    print("Pre compute complete!")

    # step 2
    gB = gmpy2.powmod(g, B, p)  #once
    gB_pow_x0 = gmpy2.mpz(1)

    for x0 in range(B):
        if gB_pow_x0 in hashtable_X1:
            x1 = hashtable_X1[gB_pow_x0]
            x = x0 * B + x1
            print("Value found!")
            print(f"x = {x}")
            return
        else:
            gB_pow_x0 = (gB_pow_x0 * gB) % p
            if x0 % 100000 == 0:
                print(f"Step 2 progress: x0 = {x0}/{B}")
    print("No value found!")
    

if __name__ == "__main__":
 
    p = 13407807929942597099574024998205846127479365820592393377723561443721764030073546976801874298166903427690031858186486050853753882811946569946433649006084171
    g = 11717829880366207009516117596335367088558084999998952205599979459063929499736583746670572176471460312928594829675428279466566527115212748467589894601965568
    h = 3239475104050450443565264378728065788649097520952449527834792452971981976143292558073856937958553180532878928001494706097394108577585732452307673444020333
    
    discrete_log(p, g, h)