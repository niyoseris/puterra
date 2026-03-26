#!/usr/bin/env python3

# EDIT THESE VALUES
PRINCIPAL = 10000
RATE = 5.5
TIME = 3
COMPOUND_FREQ = 12

def simple_interest(p, r, t):
    interest = p * (r / 100) * t
    return interest, p + interest

def compound_interest(p, r, t, n=12):
    amount = p * (1 + (r / 100) / n) ** (n * t)
    return amount - p, amount

print("=" * 50)
print("       INTEREST CALCULATOR")
print("=" * 50)

si_int, si_amt = simple_interest(PRINCIPAL, RATE, TIME)
ci_int, ci_amt = compound_interest(PRINCIPAL, RATE, TIME, COMPOUND_FREQ)

print("\nINPUT: Principal=$" + str(PRINCIPAL) + ", Rate=" + str(RATE) + "%, Time=" + str(TIME) + " years")
print("\nSIMPLE INTEREST: $" + str(round(si_int, 2)) + " earned, Total: $" + str(round(si_amt, 2)))
print("COMPOUND INTEREST: $" + str(round(ci_int, 2)) + " earned, Total: $" + str(round(ci_amt, 2)))
print("\nCompound earns $" + str(round(ci_int - si_int, 2)) + " more than simple interest!")