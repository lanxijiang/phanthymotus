#!/usr/bin/env python3
"""Opt-in actual FST/TN/G2P regression, using existing frontend dependencies.

Set VITS2_FRONTEND_TEST_ASSETS to an extracted model directory containing
frontend_data/, nltk_data/ and tn_cache/. No model inference or download runs.
An optional VITS2_TEST_PYPINYIN_PACKAGE selects an existing pure-Python package.
All generated caches and the JSON report stay in the required --output-dir.
Unknown-word neural G2P is blocked and reported as deferred by default. Opt in
to the already-installed fallback with --allow-neural-g2p; network/downloads
remain blocked in either mode. This never runs acoustic model inference.
"""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures/vits2_mixed_cases.json"
FIXTURE_SHA256 = "ac05ec49e119141493fbdeb59a2d2ecb2dbadc23647eae3a2decf7f97514da20"


def canonical(text):
    return "".join(char.casefold() for char in text if char.isalnum())


def check_semantics(case, text, normalized):
    result, norm = [], canonical(normalized)
    for phrase in case.get("contains", []):
        if canonical(phrase) not in norm:
            result.append("Missing reading: " + phrase)
    for phrase in case.get("forbidden", []):
        if canonical(phrase) in norm:
            result.append("Forbidden reading: " + phrase)
    for phrase in case.get("preserve_zh", []):
        if norm.count(canonical(phrase)) != canonical(text).count(canonical(phrase)):
            result.append("Chinese span lost or duplicated: " + phrase)
    for word in case.get("preserve_en", []):
        pattern = re.compile(r"(?<![A-Za-z])" + re.escape(word) + r"(?![A-Za-z])", re.I)
        if len(pattern.findall(normalized.replace("’", "'"))) != len(pattern.findall(text.replace("’", "'"))):
            result.append("English lexeme lost or duplicated: " + word)
    if case.get("policy") != "review" and re.search(r"[0-9]", normalized):
        result.append("Arabic digits were not verbalized")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-neural-g2p", action="store_true",
                        help="Allow the installed English phoneme fallback; still fully offline")
    args = parser.parse_args()
    assets_env = os.getenv("VITS2_FRONTEND_TEST_ASSETS")
    if not assets_env:
        parser.error("Set VITS2_FRONTEND_TEST_ASSETS; default unit tests do not require these resources")
    assets = Path(assets_env).resolve()
    if any(not (assets / part).is_dir() for part in ("frontend_data", "nltk_data", "tn_cache")):
        parser.error("Assets require frontend_data/, nltk_data/ and tn_cache/")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cache = output / "cache"
    cache.mkdir()
    tempfile.tempdir = str(cache)
    sys.dont_write_bytecode = True
    os.environ.update({"NLTK_DATA": str(assets / "nltk_data"),
                       "VITS2_FRONTEND_DATA_DIR": str(assets / "frontend_data"),
                       "VITS2_CMU_CACHE_PATH": str(cache / "cmudict_cache.pickle"),
                       "EN_TN_CACHE_DIR": str(cache / "en-tn-cache"),
                       "TMPDIR": str(cache), "OMP_NUM_THREADS": "1"})

    def offline_audit(event, values):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise RuntimeError("Network forbidden in offline frontend regression")
    sys.addaudithook(offline_audit)
    pypinyin_path = os.getenv("VITS2_TEST_PYPINYIN_PACKAGE")
    if pypinyin_path:
        package = Path(pypinyin_path)
        spec = importlib.util.spec_from_file_location("pypinyin", package / "__init__.py", submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec)
        sys.modules["pypinyin"] = module
        spec.loader.exec_module(module)
    import jieba
    jieba.dt.tmp_dir = str(cache)
    jieba.dt.cache_file = "jieba.cache"
    import nltk

    def forbidden_download(*args, **kwargs):
        raise RuntimeError("Downloads forbidden")
    nltk.download = forbidden_download
    import g2p_en

    class NeuralFallbackDeferred(RuntimeError):
        pass

    def no_neural_fallback(self, word):
        raise NeuralFallbackDeferred(word)
    if not args.allow_neural_g2p:
        g2p_en.G2p.predict = no_neural_fallback

    sys.path.insert(0, str(HERE.parent))
    from plugins.vits2_tts_trt.frontend.release_paths import configure_release_paths
    configure_release_paths(assets)
    from plugins.vits2_tts_trt.frontend.cleaner import normalize_text_mix, g2p_normalized_text_mix
    from plugins.vits2_tts_trt.frontend import cleaned_text_to_sequence_mix
    from plugins.vits2_tts_trt.frontend.chunking import iter_text_chunks
    from plugins.vits2_tts_trt.frontend.symbols import punctuation

    fixture = FIXTURE.read_bytes()
    if hashlib.sha256(fixture).hexdigest() != FIXTURE_SHA256:
        raise RuntimeError("Frozen cases changed; review expectations before updating the fixture pin")
    cases = json.loads(fixture)
    rows = []
    for case in cases:
        text = case["text"] * int(case.get("repeat", 1))
        row = {"id": case["id"], "text": text, "failures": [], "deferred_neural": False,
               "policy_review": case.get("policy") == "review", "g2p_passed": False}
        try:
            start = time.perf_counter()
            normalized = normalize_text_mix(text)
            row.update(normalized=normalized, tn_s=time.perf_counter() - start)
            row["failures"] = check_semantics(case, text, normalized)

            def get_ids(value):
                phones, tones, langs, alignment = g2p_normalized_text_mix(value)
                if not phones or "UNK" in phones or not (len(phones) == len(tones) == len(langs) == sum(alignment)):
                    raise ValueError("Missing, unknown or misaligned phonemes")
                return phones, langs, cleaned_text_to_sequence_mix(phones, tones, langs)

            phones, langs, ids = get_ids(normalized)
            row["g2p_passed"] = True
            row["ids_sha256"] = hashlib.sha256(json.dumps([list(map(int, values)) for values in ids]).encode()).hexdigest()
            languages = {language for phone, language in zip(phones, langs) if phone != "_" and phone not in punctuation}
            if case.get("preserve_zh") and "ZH" not in languages:
                row["failures"].append("Chinese text has no Chinese phonemes")
            if case.get("preserve_en") and "EN" not in languages:
                row["failures"].append("English text has no English phonemes")
            if case.get("require_multiple_chunks"):
                count = lambda value: len(get_ids(value)[2][0]) * 2 + 1
                chunks = list(iter_text_chunks(normalized, count, 256))
                row["chunks"] = [{"text": value, "tokens_with_blank": count(value)} for value in chunks]
                if len(chunks) < 2 or any(not 0 < count(value) <= 256 for value in chunks):
                    row["failures"].append("Long case violates 256-token chunk limits")
                if re.sub(r"\s+", "", "".join(chunks)) != re.sub(r"\s+", "", normalized):
                    row["failures"].append("Chunking changed text or punctuation")
        except NeuralFallbackDeferred as exc:
            row.update(deferred_neural=True, deferred_word=str(exc))
        except Exception as exc:
            row["failures"].append(f"{type(exc).__name__}: {exc}")
        row["status"] = ("failed" if row["failures"] else "deferred_neural" if row["deferred_neural"]
                         else "policy_review" if row["policy_review"] else "passed")
        rows.append(row)
    times = [row["tn_s"] for row in rows if "tn_s" in row]
    report = {"fixture_sha256": FIXTURE_SHA256, "cases": len(rows),
              "statuses": dict(Counter(row["status"] for row in rows)),
              "all_cases_validated": all(row["status"] == "passed" for row in rows),
              "audio_tested": False, "neural_g2p_enabled": args.allow_neural_g2p,
              "tn_timing_scope": "Frontend normalization only; not synthesis RTF",
              "tn_median_s": statistics.median(times) if times else None, "rows": rows}
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False, indent=2))
    return 1 if any(row["failures"] for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
