"""Minimal runner for machines without pytest: python run_tests.py"""
import importlib, sys, time, traceback
mod = importlib.import_module("tests.test_pricer")
fails = 0
for name in sorted(n for n in dir(mod) if n.startswith("test_")):
    t = time.time()
    try:
        getattr(mod, name)(); print(f"PASS {name} ({time.time()-t:.1f}s)")
    except Exception:
        fails += 1; print(f"FAIL {name}"); traceback.print_exc()
sys.exit(1 if fails else 0)
