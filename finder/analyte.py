"""
Analyte-to-product-code resolver.

Takes an analyte term (e.g. "Group A Strep") plus an optional synonym list,
text-searches openFDA classification and 510(k) records for those synonyms,
and returns the union of product codes found along with their classification
metadata.

This is explicitly heuristic. The synonym set is per-analyte and must be
reviewed by the user before trusting completeness. The output surface tells
the user what synonyms were used.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from .models import AnalyteResolution, ProductCodeInfo
from .sources.openfda import (
    get_classification_by_product_code,
    search_classification_by_term,
    search_510k_by_term,
)

# ---------------------------------------------------------------------------
# Built-in synonym sets for common analytes.
# Extend this dict to add new analytes without touching engine code.
# ---------------------------------------------------------------------------

# Supplemental product codes per analyte.
# These are three-letter codes from openFDA that cover multiplex panels or
# systems that detect the analyte but whose device names don't mention it,
# making them invisible to synonym text search.
# Product codes are from openFDA; the analyte<->product-code association is
# curated knowledge and must be reviewed/updated per-study.
SUPPLEMENTAL_PRODUCT_CODES: dict[str, list[str]] = {
    "group a strep": [
        "PEN",  # 866.3365 — multiplex blood-culture ID panels (e.g. BioFire BCID2)
    ],
}


BUILTIN_SYNONYMS: dict[str, list[str]] = {
    "group a strep": [
        "Group A Strep",
        "Group A Streptococcus",
        "Streptococcus pyogenes",
        "GAS",
        "S. pyogenes",
        "Strep A",
    ],
    "group b strep": [
        "Group B Strep",
        "Group B Streptococcus",
        "Streptococcus agalactiae",
        "GBS",
    ],
    "influenza a": [
        "Influenza A",
        "Flu A",
        "Influenza type A",
    ],
    "sars-cov-2": [
        "SARS-CoV-2",
        "COVID-19",
        "coronavirus 2019",
        "2019-nCoV",
    ],
    "rsv": [
        "RSV",
        "respiratory syncytial virus",
        "Respiratory Syncytial Virus",
    ],
}


def _normalize(term: str) -> str:
    return term.lower().strip()


def get_synonyms(analyte_term: str, extra_synonyms: Optional[list[str]] = None) -> list[str]:
    """Return synonym set for analyte_term, including any caller-supplied extras."""
    key = _normalize(analyte_term)
    synonyms = list(BUILTIN_SYNONYMS.get(key, [analyte_term]))
    if extra_synonyms:
        for s in extra_synonyms:
            if s not in synonyms:
                synonyms.append(s)
    return synonyms


def _extract_product_code_info(cls_record: dict) -> Optional[ProductCodeInfo]:
    pc = cls_record.get("product_code")
    if not pc:
        return None
    return ProductCodeInfo(
        product_code=pc,
        device_name=cls_record.get("device_name", ""),
        regulation_number=cls_record.get("regulation_number"),
        device_class=cls_record.get("device_class"),
        medical_specialty=cls_record.get("medical_specialty_description"),
        definition=cls_record.get("definition"),
    )


def resolve_analyte(
    analyte_term: str,
    extra_synonyms: Optional[list[str]] = None,
    medical_specialty: Optional[str] = None,
) -> AnalyteResolution:
    """
    Resolve an analyte term to a set of product codes via synonym text search.

    Steps:
    1. Build the synonym set.
    2. For each synonym, text-search classification records.
    3. For each synonym, text-search 510(k) device names to pick up product
       codes not found via classification search alone.
    4. For each unique product code found, fetch its canonical classification
       metadata.
    5. Return the union.
    """
    synonyms = get_synonyms(analyte_term, extra_synonyms)
    key = _normalize(analyte_term)
    supplemental_pcs = SUPPLEMENTAL_PRODUCT_CODES.get(key, [])

    # Collect product codes -> best classification record seen
    pc_map: dict[str, ProductCodeInfo] = {}

    # Step 2: classification text search.
    # Skip very short synonyms (< 6 chars) in broad classification search to
    # avoid matching unrelated devices (e.g. "GAS" matches many device names).
    for syn in synonyms:
        if len(syn) < 6:
            continue
        for rec in search_classification_by_term(syn):
            # Optionally filter by medical specialty (e.g. "MN" for microbiology)
            if medical_specialty and rec.get("medical_specialty_description", "").upper() != medical_specialty.upper():
                continue
            info = _extract_product_code_info(rec)
            if info and info.product_code not in pc_map:
                pc_map[info.product_code] = info

    # Step 3: 510(k) text search to catch additional product codes.
    # Require longer synonyms here too to limit false positives.
    for syn in synonyms:
        if len(syn) < 6:
            continue
        for rec in search_510k_by_term(syn):
            pc = rec.get("product_code")
            if pc and pc not in pc_map:
                # Fetch classification metadata for this product code
                cls_records = get_classification_by_product_code(pc)
                if cls_records:
                    info = _extract_product_code_info(cls_records[0])
                    if info:
                        pc_map[pc] = info
                else:
                    # Use a stub if classification lookup returns nothing
                    pc_map[pc] = ProductCodeInfo(
                        product_code=pc,
                        device_name=rec.get("device_name", ""),
                    )

    # Step 4: add curated supplemental product codes (panels whose device names
    # don't mention the analyte, so synonym search can't find them).
    for pc in supplemental_pcs:
        if pc not in pc_map:
            cls_records = get_classification_by_product_code(pc)
            if cls_records:
                info = _extract_product_code_info(cls_records[0])
                if info:
                    info = info.model_copy(
                        update={"definition": (info.definition or "") + " [supplemental: curated panel association]"}
                    )
                    pc_map[pc] = info
            else:
                pc_map[pc] = ProductCodeInfo(
                    product_code=pc,
                    device_name="[curated panel supplement — classification not found]",
                )

    return AnalyteResolution(
        analyte_term=analyte_term,
        synonyms_used=synonyms,
        product_codes=list(pc_map.values()),
    )
