"""Minimal runner for machines without pytest: python run_tests.py [module ...]"""
import importlib, sys, time, traceback
fails = 0
for modname in sys.argv[1:] or ["tests.test_pricer", "tests.test_flow"]:
    mod = importlib.import_module(modname)
    for name in sorted(n for n in dir(mod) if n.startswith("test_")):
        t = time.time()
        try:
            getattr(mod, name)(); print(f"PASS {name} ({time.time()-t:.1f}s)")
        except Exception:
            fails += 1; print(f"FAIL {name}"); traceback.print_exc()
sys.exit(1 if fails else 0)
