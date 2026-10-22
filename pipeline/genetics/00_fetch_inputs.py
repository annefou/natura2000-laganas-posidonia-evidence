"""Fetch the open genetic inputs for the Posidonia oceanica genetic-vulnerability prototype (Greece).

Usage: pixi run python pipeline/genetics/00_fetch_inputs.py <ROOT>      (ROOT = work/genetics)

Inputs (all open, no account):
1. Litsi-Mizan V, Garcia-Escudero CA, Tsigenopoulos CS, Tsiaras K, Gerakaris V, Apostolaki ET (2024).
   Unravelling the genetic pattern of seagrass (Posidonia oceanica) meadows in the Eastern Mediterranean Sea.
   Biodiversity and Conservation 33:257-280. doi:10.1007/s10531-023-02746-0. University of Crete / HCMR.
   Licence: CC BY 4.0 (article and electronic supplementary material, Crossref licence record).
   - ESM1 (xlsx): multilocus genotypes, 12 microsatellite loci, 334 MLGs retained after clone removal, 18 meadows.
     https://static-content.springer.com/esm/art%3A10.1007%2Fs10531-023-02746-0/MediaObjects/10531_2023_2746_MOESM1_ESM.xlsx
   - ESM2 (docx): Table S1 site coordinates (2 decimals) and sampling depth; Table S2 primers.
     https://static-content.springer.com/esm/art%3A10.1007%2Fs10531-023-02746-0/MediaObjects/10531_2023_2746_MOESM2_ESM.docx
   - Table 1 of the article (ramets genotyped N per meadow) is not in the ESM; it is transcribed in
     pipeline/genetics/litsi_mizan_2024_table1.csv (with the checks used to validate the transcription).

2. Natural Earth 1:10m land polygons (basemap for the figure only). Public domain.
   https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip

Not fetched automatically (documented in REPORT.txt):
- Jahnke M, Olsen JL, Procaccini G (2015) Mol Ecol 24:2336-2348, doi:10.1111/mec.13174; Dryad doi:10.5061/dryad.744fb,
  CC0, 56 meadows Mediterranean-wide (per-meadow summary xlsx). Dryad file downloads need a browser (bot challenge)
  or a Dryad API token; if you have the file, put it in <ROOT>/data/inputs/jahnke2015_genotype_threat.xlsx.

Outputs: <ROOT>/data/inputs/{litsi_mizan_2024_esm1_genotypes.xlsx, litsi_mizan_2024_esm2.docx, ne_10m_land.zip, sites.csv, sources.json}
"""
import sys, os, re, json, hashlib, zipfile, urllib.request
import pandas as pd

ROOT = sys.argv[1]
OUT = f"{ROOT}/data/inputs"
os.makedirs(OUT, exist_ok=True)
BASE = "https://static-content.springer.com/esm/art%3A10.1007%2Fs10531-023-02746-0/MediaObjects/"
FILES = {"litsi_mizan_2024_esm1_genotypes.xlsx": BASE + "10531_2023_2746_MOESM1_ESM.xlsx",
         "litsi_mizan_2024_esm2.docx": BASE + "10531_2023_2746_MOESM2_ESM.docx"}
src = []
NE = ("ne_10m_land.zip", "https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip")
for fn, url in list(FILES.items()) + [NE]:
    p = f"{OUT}/{fn}"
    if not os.path.exists(p):
        with urllib.request.urlopen(url, timeout=60) as r, open(p, "wb") as f:
            f.write(r.read())
    src.append(dict(file=fn, url=url, sha256=hashlib.sha256(open(p, "rb").read()).hexdigest(),
                    licence="public domain" if fn == NE[0] else "CC BY 4.0",
                    doi=None if fn == NE[0] else "10.1007/s10531-023-02746-0"))
    print("ok", fn)

# Table S1 from ESM2 (docx = zip of XML; parse the first table without extra dependencies)
xml = zipfile.ZipFile(f"{OUT}/litsi_mizan_2024_esm2.docx").read("word/document.xml").decode("utf8")
tbl = re.search(r"<w:tbl>.*?</w:tbl>", xml, re.S).group(0)
rows = []
for tr in re.findall(r"<w:tr[ >].*?</w:tr>", tbl, re.S):
    cells = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", tc)).strip() for tc in re.findall(r"<w:tc>.*?</w:tc>", tr, re.S)]
    rows.append(cells)
hdr = rows[0]
assert hdr[:7] == ["Sea", "Region", "Site", "Site code", "LAT", "LON", "Depth (m)"], hdr
s = pd.DataFrame(rows[1:], columns=hdr).rename(columns={"Sea": "sea", "Region": "region", "Site": "site", "Site code": "site_code",
                                                       "LAT": "lat", "LON": "lon", "Depth (m)": "depth_m"})
s[["sea", "region"]] = s[["sea", "region"]].replace("", None).ffill()
for c in ("lat", "lon", "depth_m"):
    s[c] = s[c].astype(float)
assert len(s) == 18, len(s)
s.to_csv(f"{OUT}/sites.csv", index=False)
json.dump(src, open(f"{OUT}/sources.json", "w"), indent=1)
print(s.to_string())
