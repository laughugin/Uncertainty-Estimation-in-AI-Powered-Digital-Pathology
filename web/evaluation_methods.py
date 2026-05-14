from __future__ import annotations

from typing import Any


METHOD_UI: dict[str, dict[str, Any]] = {
    "confidence": {
        "id": "confidence",
        "name": "Confidence / MSP",
        "short_name": "Confidence",
        "description": "Maximum softmax probability baseline used as the simplest confidence estimate.",
        "overview": "Use this as the baseline method. It shows raw predictive quality and how overconfident a plain softmax score can be.",
        "evaluation_blocks": [
            {
                "title": "Predictive performance",
                "text": "Shows the plain classification quality of the softmax baseline before any stronger uncertainty method is compared against it.",
            },
            {
                "title": "Calibration",
                "text": "Shows whether softmax confidence is numerically trustworthy or overconfident on correct and incorrect predictions.",
            },
            {
                "title": "Uncertainty quality",
                "text": "Checks whether simple confidence can separate errors from correct predictions. This is usually the weakest point of the baseline.",
            },
            {
                "title": "Selective prediction",
                "text": "Shows whether rejecting low-confidence cases lowers risk in a meaningful way, even for a crude baseline.",
            },
            {
                "title": "Pathology proxy reporting",
                "text": "Gives a pseudo-slide summary on PCAM so the baseline can still be interpreted in a pathology-style reporting frame.",
            },
            {
                "title": "Shift / OOD robustness",
                "text": "Shows how a simple baseline reacts under blur, JPEG, color, and noise shift before comparing stronger uncertainty methods.",
            },
        ],
        "notes": [
            "Treat this as a weak uncertainty baseline, not as a strong epistemic method.",
            "If calibration or OOD behavior is poor here, that is expected and useful as a comparison point.",
        ],
        "foundation_reference_keys": [],
    },
    "temperature_scaled": {
        "id": "temperature_scaled",
        "name": "Confidence + Temperature Scaling",
        "short_name": "Temp Scaled",
        "description": "Confidence baseline with post-hoc temperature scaling fitted on a held-out split.",
        "overview": "Use this as the calibrated baseline. It shows how much plain softmax confidence improves after a standard post-hoc calibration step.",
        "evaluation_blocks": [
            {
                "title": "Predictive performance",
                "text": "Checks whether calibration leaves ranking metrics essentially unchanged while preserving ordinary classification quality.",
            },
            {
                "title": "Calibration",
                "text": "This is the main comparison point for whether simple post-hoc calibration is enough before moving to stronger uncertainty methods.",
            },
            {
                "title": "Uncertainty quality",
                "text": "Checks whether calibrated confidence separates errors better than raw softmax confidence, while remaining cheaper than stochastic methods.",
            },
            {
                "title": "Selective prediction",
                "text": "Shows whether a calibrated baseline supports more reliable high-confidence reporting than raw confidence alone.",
            },
            {
                "title": "Pathology proxy reporting",
                "text": "Provides the same pseudo-slide reporting view, now with calibrated confidence scores.",
            },
            {
                "title": "Shift / OOD robustness",
                "text": "Useful as a reference because temperature scaling often improves in-distribution calibration more than OOD robustness.",
            },
        ],
        "notes": [
            "This is a calibrated baseline, not a richer epistemic uncertainty method.",
            "Strong in-distribution calibration here does not necessarily imply strong OOD behavior.",
        ],
        "foundation_reference_keys": ["guo2017calibration"],
    },
    "mc_dropout": {
        "id": "mc_dropout",
        "name": "MC Dropout",
        "short_name": "MC Dropout",
        "description": "Stochastic test-time dropout with multiple forward passes to approximate Bayesian uncertainty.",
        "overview": "Use this to inspect whether stochastic dropout improves calibration, error detection, and selective deferral over plain confidence.",
        "evaluation_blocks": [
            {
                "title": "Predictive performance",
                "text": "Checks whether the stochastic method preserves or improves ordinary classification quality relative to the baseline.",
            },
            {
                "title": "Calibration",
                "text": "Checks whether averaging multiple dropout passes makes the reported confidence align better with observed correctness.",
            },
            {
                "title": "Uncertainty quality",
                "text": "This is the main place to test whether MC Dropout uncertainty rises on mistakes more reliably than plain softmax confidence.",
            },
            {
                "title": "Selective prediction",
                "text": "Shows whether dropout-based uncertainty can support safer deferral or abstention on difficult cases.",
            },
            {
                "title": "Pathology proxy reporting",
                "text": "Provides the same pseudo-slide reporting view, now using the stochastic uncertainty-aware predictions.",
            },
            {
                "title": "Shift / OOD robustness",
                "text": "Checks whether predictive uncertainty rises under shift and whether that rise helps discriminate ID from shifted samples.",
            },
        ],
        "notes": [
            "This is the main approximate Bayesian baseline in the project.",
            "Interpret gains across calibration, error detection, and selective prediction together rather than from one metric alone.",
        ],
        "foundation_reference_keys": ["gal2016dropout"],
    },
    "deep_ensemble": {
        "id": "deep_ensemble",
        "name": "Deep Ensemble",
        "short_name": "Deep Ensemble",
        "description": "Average prediction from multiple independently trained models.",
        "overview": "Use this as the strongest empirical uncertainty baseline for comparing predictive quality, calibration, and selective prediction.",
        "evaluation_blocks": [
            {
                "title": "Predictive performance",
                "text": "Checks whether averaging multiple independently trained models improves ordinary prediction quality.",
            },
            {
                "title": "Calibration",
                "text": "Tests whether ensemble averaging produces more reliable confidence than a single model.",
            },
            {
                "title": "Uncertainty quality",
                "text": "Shows whether disagreement across members leads to uncertainty that better marks likely errors.",
            },
            {
                "title": "Selective prediction",
                "text": "Shows whether ensemble uncertainty supports safer deferral with better risk-coverage behavior.",
            },
            {
                "title": "Pathology proxy reporting",
                "text": "Provides the pseudo-slide reporting view using the ensemble-averaged predictions.",
            },
            {
                "title": "Shift / OOD robustness",
                "text": "Use this especially for shift and OOD analysis because digital pathology literature often reports stronger robustness for ensembles under dataset shift.",
            },
        ],
        "notes": [
            "This is usually the strongest practical baseline in uncertainty benchmarks.",
            "Its main value is robustness under shift, not only raw accuracy.",
        ],
        "foundation_reference_keys": ["lakshminarayanan2017simple"],
    },
}


def _format_numbered_reference(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "numbered",
        "label": f"[{entry.get('number')}] {entry.get('short')}",
        "citation": entry.get("citation"),
        "summary": entry.get("summary"),
        "number": entry.get("number"),
        "key": entry.get("key"),
    }


def _format_foundation_reference(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "foundation",
        "label": entry.get("label"),
        "citation": entry.get("citation"),
        "summary": None,
        "number": None,
        "key": entry.get("key"),
    }


def build_evaluation_methods(reference_catalog: dict[str, Any]) -> list[dict[str, Any]]:
    ordered = {item.get("number"): item for item in reference_catalog.get("ordered_literature", [])}
    foundation = {item.get("key"): item for item in reference_catalog.get("foundation_references", [])}
    method_map = reference_catalog.get("method_reference_map", {}) or {}
    methods: list[dict[str, Any]] = []

    for method_id in ("confidence", "temperature_scaled", "mc_dropout", "deep_ensemble"):
        meta = dict(METHOD_UI[method_id])
        numbered_refs = [
            _format_numbered_reference(ordered[number])
            for number in method_map.get(method_id, [])
            if number in ordered
        ]
        foundation_refs = [
            _format_foundation_reference(foundation[key])
            for key in meta.get("foundation_reference_keys", [])
            if key in foundation
        ]
        meta["scientific_references"] = numbered_refs + foundation_refs
        methods.append(meta)
    return methods


def build_evaluation_method_map(reference_catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in build_evaluation_methods(reference_catalog)}
