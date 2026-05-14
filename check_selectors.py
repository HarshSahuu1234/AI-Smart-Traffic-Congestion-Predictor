"""Quick script to check what Streamlit renders for sidebar toggle."""
import requests
import re

r = requests.get("http://localhost:8501")
html = r.text

# Find all data-testid attributes
testids = re.findall(r'data-testid="([^"]*)"', html)
for t in sorted(set(testids)):
    if any(kw in t.lower() for kw in ['side', 'collaps', 'expand', 'toggle', 'nav', 'header', 'arrow']):
        print(f"  MATCH: {t}")

print(f"\nAll testids ({len(set(testids))}):")
for t in sorted(set(testids)):
    print(f"  {t}")
