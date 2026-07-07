from devices.FracMemCap import BiolekMemcapacitor

# --- Endpoint check: fresh device, q=0 ---
b = BiolekMemcapacitor()

b.program(1)
assert b.x == 0.0, f"program(1) should give x=0.0, got {b.x}"
g1 = b.current_conductance(1)
print("program(1): x=", b.x, "G=", g1)

b.program(0)
assert b.x == 1.0, f"program(0) should give x=1.0, got {b.x}"
g0 = b.current_conductance(1)
print("program(0): x=", b.x, "G=", g0)

assert g1 > g0, f"expected program(1) conductance > program(0), got G1={g1}, G0={g0}"

b.program(0.5)
print("program(0.5): x=", b.x, "current_conductance(1)=", b.current_conductance(1))

# --- Offset check with nonzero charge ---
b2 = BiolekMemcapacitor(IC=1e-9)  # nonzero initial charge
b2.program(1)
offset = b2.current_offset(1)
print("nonzero-q current_offset(1)=", offset)
assert offset != 0.0, "expected nonzero offset with nonzero q, got 0 — check whether program() resets q"

# also check: does program() zero out q as a side effect?
print("q after program() with IC=1e-9:", b2.q)