#!/bin/bash
# Median of N Lighthouse runs against a URL. One run of a live site is not a measurement.
URL=$1; N=${2:-3}; OUT=/tmp/claude-501/-Users-theojohn-dev-enargeia/b988e848-6b32-4a21-a668-f572ba22b961/scratchpad
for i in $(seq 1 $N); do
  npx --yes lighthouse "$URL" --quiet --chrome-flags="--headless=new --no-sandbox" \
    --output=json --output-path=$OUT/lh-run$i.json > /dev/null 2>&1
done
python3 - "$OUT" "$N" <<'PY'
import json,sys,statistics
out,n=sys.argv[1],int(sys.argv[2])
keys=['performance']
metrics=['first-contentful-paint','largest-contentful-paint','total-blocking-time','speed-index']
rows={k:[] for k in keys+metrics}
for i in range(1,n+1):
    d=json.load(open(f'{out}/lh-run{i}.json'))
    rows['performance'].append(round(d['categories']['performance']['score']*100))
    for m in metrics: rows[m].append(d['audits'][m]['numericValue'])
print(f"  performance  median {statistics.median(rows['performance']):.0f}   runs {rows['performance']}")
for m in metrics:
    print(f"  {m:24} median {statistics.median(rows[m]):7.0f} ms   runs {[round(x) for x in rows[m]]}")
PY
