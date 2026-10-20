#!/usr/bin/env bash
# Fetch the opacity data the TauREx-III forward models and retrievals read (README, "Data"): ExoMol cross-sections
# into DATA/XSEC (_CONFIG.CROSS_SECTION_PATH) and HITRAN CIA into DATA/CIA (_CONFIG.CIA_PATH). About 4 GB.
# Files already there are skipped. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/DATA"

get() {  # get <folder> <local name> <url>
    if [ -e "$1/$2" ]; then echo "have $1/$2"; return; fi
    echo "get  $1/$2"
    curl -#fL --retry 3 -o "$1/$2.part" "$3" && mv "$1/$2.part" "$1/$2"
}

X=https://www.exomol.com/db
R=R15000_0.3-50mu.xsec.TauREx.h5
get XSEC "1H2-16O__POKAZATEL.$R"   "$X/H2O/1H2-16O/POKAZATEL/1H2-16O__POKAZATEL__$R"
get XSEC "12C-1H4__YT34to10.$R"    "$X/CH4/12C-1H4/YT34to10/12C-1H4__YT34to10.$R"
get XSEC "12C-16O__Li2015.$R"      "$X/CO/12C-16O/Li2015/12C-16O__Li2015.$R"
get XSEC "14N-1H3__CoYuTe.$R"      "$X/NH3/14N-1H3/CoYuTe/14N-1H3__CoYuTe.$R"
get XSEC "32S-16O2__ExoAmes.$R"    "$X/SO2/32S-16O2/ExoAmes/32S-16O2__ExoAmes.$R"
get XSEC "12C2-1H2__aCeTY.$R"      "$X/C2H2/12C2-1H2/aCeTY/12C2-1H2__aCeTY.$R"
get XSEC "12C2-1H4__MaYTY.$R"      "$X/C2H4/12C2-1H4/MaYTY/12C2-1H4__MaYTY.$R"
get XSEC "1H2-32S__AYT2.$R"        "$X/H2S/1H2-32S/AYT2/1H2-32S__AYT2.$R"
get XSEC "1H-12C-14N__Harris.$R"   "$X/HCN/1H-12C-14N/Harris/1H-12C-14N__Harris.$R"
# the paper uses 12C-16O2__HITEMP, which ExoMol no longer serves; UCL-4000 stands in unless a CO2 file is already here
if ! ls XSEC/12C-16O2__* >/dev/null 2>&1; then
    echo "note: CO2 from UCL-4000, not the paper's HITEMP (README, \"Data\")"
    get XSEC "12C-16O2__UCL-4000.$R" "$X/CO2/12C-16O2/UCL-4000/12C-16O2__UCL-4000.$R"
fi

# HITRAN 2011 CIA; skipped where a TauREx .db of the same pair is already here
for pair in H2-H2 H2-He; do
    [ -e "CIA/$pair.db" ] || get CIA "${pair}_2011.cia" "https://hitran.org/data/CIA/main/${pair}_2011.cia"
done
