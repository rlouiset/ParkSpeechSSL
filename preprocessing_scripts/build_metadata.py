"""
Build derivatives/metadata.csv: one row per individual (dataset, patient_hash)
with whatever age / sex / severity the raw datasets provide, for the clinical
probes in pdspeech_ssl/clinical_probe.py.

patient_hash is recomputed exactly as each preprocess_*.py script does, so rows
join onto the derivative filenames ({LABEL}_{patient_hash}_{idx}_{Dataset}.wav)
directly. Every row is checked against the derivatives folder and unmatched
rows are reported (and dropped) rather than silently written.

Sources:
- NeuroVoz: data/metadata/metadata_{pd,hc}.csv -- age, sex, UPDRS (total, PD
  only), H&Y, years since diagnosis. Sex is coded 1/0; the dataset doesn't
  document which is which, so it's written as-is (it's only used as a binary
  probe target, so the direction doesn't matter).
- KCL (MDVR-KCL): no metadata file, but the filename's three trailing digits
  are H&Y, UPDRS II-5 and UPDRS III-18 ({id}_{hc|pd}_{hy}_{u2_5}_{u3_18}).
  Not consistent across a patient's ReadText/SpontaneousDialogue files is
  treated as an error.
- IPVS: age/sex in "15 YHC.xlsx", "Tab 3.xlsx", "TAB 5.xlsx", keyed by
  name + surname initial, matched against the patient folder names (some
  folders are surname-first, e.g. "LISCO G", so both orders are tried).
- FredPrior: Demographics_age_sex.xlsx, keyed by the full sample id.
CzechPD, SJTU, YeTalkBank, RAWDysPeech have none of these and are skipped.

Columns: dataset, label, patient_hash, age, sex (M/F), updrs, hy, years_dx
(empty = unknown). updrs is NeuroVoz's total UPDRS; KCL's speech sub-items are
not comparable to it and are not written.
"""

import hashlib
import re
import unicodedata
from pathlib import Path

import pandas as pd

RAW = Path("/Users/robinlouiset/Documents/ParkSpeechData/raw")
DERIVATIVES = Path("/Users/robinlouiset/Documents/ParkSpeechData/derivatives")
OUT_PATH = DERIVATIVES / "metadata.csv"

COLUMNS = ["dataset", "label", "patient_hash", "age", "sex", "updrs", "hy", "years_dx"]


def md5_8(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:8]


def neurovoz() -> list[dict]:
    rows = []
    for group in ("pd", "hc"):
        meta = pd.read_csv(RAW / "NeuroVoz/data/metadata" / f"metadata_{group}.csv").drop_duplicates("ID")
        label = group.upper()
        for _, r in meta.iterrows():
            # filenames carry the id zero-padded to 4 digits, e.g. HC_A1_0034.wav
            pid = f"{int(r['ID']):04d}"
            sex = {1: "M", 0: "F"}.get(r["Sex"]) if pd.notna(r["Sex"]) else None
            rows.append({
                "dataset": "NeuroVoz", "label": label, "patient_hash": md5_8(f"{label}::{pid}"),
                "age": r["Age"], "sex": sex,
                "updrs": r["UPDRS scale"] if label == "PD" else None,
                "hy": r["H-Y Stadium"] if label == "PD" else 0.0,
                "years_dx": r["Time Disease (years)"] if label == "PD" else None,
            })
    return rows


KCL_RE = re.compile(r"^(ID\d+)_?(hc|pd)_(\d+)_(\d+)_(\d+)$", re.IGNORECASE)


def kcl() -> list[dict]:
    hy_by_patient: dict[tuple[str, str], str] = {}
    for wav_path in sorted((RAW / "26-29_09_2017_KCL").glob("*/*/*.wav")):
        m = KCL_RE.match(wav_path.stem)
        label, pid = wav_path.parent.name, m.group(1).upper()
        hy = m.group(3)
        prev = hy_by_patient.setdefault((label, pid), hy)
        if prev != hy:
            raise ValueError(f"KCL {pid}: inconsistent H&Y across files ({prev} vs {hy})")
    return [
        {"dataset": "KCL", "label": label, "patient_hash": md5_8(f"{label}::{pid}"), "hy": float(hy)}
        for (label, pid), hy in hy_by_patient.items()
    ]


IPVS_ROOT = RAW / "Italian Parkinson's Voice and speech"
# (group folder, demographics file, label)
IPVS_GROUPS = [
    ("15 Young Healthy Control", "15 YHC.xlsx", "HC"),
    ("22 Elderly Healthy Control", "Tab 3.xlsx", "HC"),
    ("28 People with Parkinson's disease", "TAB 5.xlsx", "PD"),
]
# spreadsheet "NAME S" -> folder, where the folder is "SURNAME N" (checked by hand: the
# only folder with that surname initial + name initial). Ambiguous ones are left out on
# purpose: PORCELLI A (Antonio P or Antonella P?), Mario B (spreadsheet says "Mario M").
IPVS_ALIASES = {"GIUSEPPE L": "LISCO G", "LUIGIA S": "SUMMO L"}


def _ipvs_patient_folders(group_dir: Path) -> set[str]:
    # PD patients sit one level deeper, under "x-y" range folders
    dirs = [d for d in group_dir.iterdir() if d.is_dir()]
    if all(re.fullmatch(r"\d+-\d+", d.name) for d in dirs):
        dirs = [p for d in dirs for p in d.iterdir() if p.is_dir()]
    return {d.name for d in dirs}


def _ascii_upper(text) -> str:
    """"Nicolo'" (spreadsheet) and "Nicolò" (folder) -> "NICOLO"; also drops the "*" markers."""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[*']", "", text).strip().upper()


def ipvs() -> list[dict]:
    rows = []
    for group, xlsx, label in IPVS_GROUPS:
        group_dir = IPVS_ROOT / group
        folders = {_ascii_upper(name): name for name in _ipvs_patient_folders(group_dir)}
        sheet = pd.read_excel(group_dir / xlsx, header=None)
        header_row = sheet.index[sheet.apply(lambda r: "name" in r.astype(str).str.strip().str.lower().tolist(), axis=1)][0]
        sheet.columns = [str(c).strip().lower() for c in sheet.iloc[header_row]]
        sheet = sheet.iloc[header_row + 1:].dropna(subset=["name"])
        # footnote rows ("registrazioni di scarsa qualità", ...) have no age
        sheet = sheet[pd.to_numeric(sheet["age"], errors="coerce").notna()]
        for _, r in sheet.iterrows():
            name = _ascii_upper(r["name"])
            surname = _ascii_upper(r["surname"])
            key = IPVS_ALIASES.get(f"{name} {surname}", f"{name} {surname}")
            folder = folders.get(key) or folders.get(f"{surname} {name}")
            if folder is None:
                print(f"[IPVS] no folder for {group}: {name} {surname}")
                continue
            # PD patients appearing in several range folders are listed once per folder
            # (Vito S even with ages 71/71/70) -- the first row wins, see drop_duplicates in main
            rows.append({
                "dataset": "IPVS", "label": label, "patient_hash": md5_8(f"{group}::{folder.strip().upper()}"),
                "age": r["age"], "sex": str(r["sex"]).strip().upper(),
            })
    return rows


FRED_HC_RE = re.compile(r"^AH_([A-Za-z0-9]+)_[0-9A-F-]+$", re.IGNORECASE)
FRED_PD_RE = re.compile(r"^AH_(\d+)-[0-9A-F-]+$", re.IGNORECASE)


def fredprior() -> list[dict]:
    meta = pd.read_excel(RAW / "FredPrior_AnuHIER_2023_SciRep/Demographics_age_sex.xlsx", sheet_name="Parselmouth")
    rows = []
    for _, r in meta.iterrows():
        label = {"PwPD": "PD"}.get(str(r["Label"]).strip(), str(r["Label"]).strip())
        m = (FRED_HC_RE if label == "HC" else FRED_PD_RE).match(str(r["Sample ID"]).strip())
        if m is None:
            print(f"[FredPrior] unparsable sample id: {r['Sample ID']}")
            continue
        rows.append({
            "dataset": "FredPrior", "label": label, "patient_hash": md5_8(f"{label}::{m.group(1)}"),
            "age": r["Age"], "sex": str(r["Sex"]).strip().upper(),
        })
    return rows


def existing_individuals() -> set[tuple[str, str, str]]:
    keys = set()
    for wav_path in DERIVATIVES.glob("*/*.wav"):
        label, phash = wav_path.stem.split("_")[:2]
        keys.add((wav_path.parent.name, label, phash))
    return keys


def main() -> None:
    df = pd.DataFrame(neurovoz() + kcl() + ipvs() + fredprior(), columns=COLUMNS)
    df = df.drop_duplicates(["dataset", "label", "patient_hash"])
    existing = existing_individuals()
    found = df.apply(lambda r: (r["dataset"], r["label"], r["patient_hash"]) in existing, axis=1)
    for _, r in df[~found].iterrows():
        print(f"[missing in derivatives] {r['dataset']} {r['label']} {r['patient_hash']}")
    df = df[found]
    df.to_csv(OUT_PATH, index=False)

    print(f"wrote {len(df)} individuals to {OUT_PATH}")
    summary = df.groupby(["dataset", "label"]).agg(
        n=("patient_hash", "size"), age=("age", "count"), sex=("sex", "count"), updrs=("updrs", "count"), hy=("hy", "count"),
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
