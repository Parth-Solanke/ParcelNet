import sys
sys.path.insert(0, ".")
from fastapi.testclient import TestClient
from backend.app.main import app  # noqa

client = TestClient(app)
print("=== API SMOKE TEST (post-fix) ===")
tests = [
    ("GET",  "/health"),
    ("GET",  "/projects"),
    ("GET",  "/projects/demo_project/layers/parcels"),
    ("GET",  "/projects/demo_project/layers/buildings"),
    ("GET",  "/projects/demo_project/layers/roads"),
    ("GET",  "/projects/demo_project/validation/issues"),
    ("POST", "/projects/demo_project/validation/run"),
    ("POST", "/projects/demo_project/validation/autofix"),
    ("GET",  "/projects/demo_project/gt/verification-list"),
    ("GET",  "/projects/demo_project/layers/export/download?format=geojson&layer=parcels"),
    ("PATCH","/parcels/p001"),
]

all_ok = True
for item in tests:
    method, url = item
    if method == "GET":
        r = client.get(url)
    elif method == "POST":
        r = client.post(url, json={})
    elif method == "PATCH":
        r = client.patch(url, json={"version": 1, "status": "verified"})
    ok = r.status_code < 400
    if not ok:
        all_ok = False
    tag = "OK  " if ok else "FAIL"
    print(f"  {tag} [{r.status_code}] {method} {url[:65]}")
    if not ok:
        print(f"       > {r.text[:300]}")

print()
print("RESULT:", "ALL PASS" if all_ok else "SOME FAILURES")
