"""
Tests para metricas con referencia y TextNormalizer.

Cobertura:
  - TextNormalizer: normalize, _dashes_to_spaces, _remove_accents, tokenize
  - ReferenceBasedMetrics.f1_score: overlap, empty, identico, cero, parcial, duplicados, dashes
  - ReferenceBasedMetrics.exact_match: identico, normalizacion, acentos, puntuacion, dashes, empty
  - ReferenceBasedMetrics.accuracy: yes/no, case insensitive, valid_labels, empty

Sin dependencias externas (no requiere NIM, mocks, ni infra).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.metrics import (
    TextNormalizer,
    ReferenceBasedMetrics,
    MetricType,
)


# =================================================================
# SECCION 1: TextNormalizer
# =================================================================

def test_normalize_basic():
    """Lowercase + strip + espacio normalizado."""
    assert TextNormalizer.normalize("  Hello   World  ") == "hello world"
    print("PASS: normalize basico (lowercase, espacios)")


def test_normalize_empty():
    assert TextNormalizer.normalize("") == ""
    assert TextNormalizer.normalize(None) == ""  # falsy
    print("PASS: normalize con input vacio/None")


def test_remove_accents():
    """NFD decomposition elimina diacriticos."""
    assert TextNormalizer._remove_accents("café") == "cafe"
    assert TextNormalizer._remove_accents("über") == "uber"
    assert TextNormalizer._remove_accents("niño") == "nino"
    assert TextNormalizer._remove_accents("résumé") == "resume"
    # Sin acentos no cambia
    assert TextNormalizer._remove_accents("hello") == "hello"
    print("PASS: _remove_accents (cafe, uber, nino, resume)")


def test_dashes_to_spaces():
    """Todas las variantes de dash se convierten a espacio."""
    # Hyphen-minus (U+002D) - el mas comun
    assert "1969 1974" == TextNormalizer._dashes_to_spaces("1969-1974").strip()
    # En-dash (U+2013)
    assert "1969 1974" == TextNormalizer._dashes_to_spaces("1969\u20131974").strip()
    # Em-dash (U+2014)
    assert "1969 1974" == TextNormalizer._dashes_to_spaces("1969\u20141974").strip()
    # Minus sign (U+2212)
    assert "1969 1974" == TextNormalizer._dashes_to_spaces("1969\u22121974").strip()
    # Sin dashes no cambia
    assert TextNormalizer._dashes_to_spaces("hello world") == "hello world"
    print("PASS: _dashes_to_spaces (hyphen, en-dash, em-dash, minus sign)")


def test_normalize_with_dashes_and_punctuation():
    """Caso real: '1969-1974' normaliza a '1969 1974' (dos tokens)."""
    result = TextNormalizer.normalize("1969-1974")
    assert result == "1969 1974", f"Esperado '1969 1974', obtenido '{result}'"

    tokens = TextNormalizer.tokenize("1969-1974")
    assert tokens == ["1969", "1974"], f"Esperado 2 tokens, obtenido {tokens}"
    print("PASS: normalize con dashes produce tokens separados")


def test_tokenize_punctuation_only():
    """String con solo puntuacion produce lista vacia."""
    tokens = TextNormalizer.tokenize("!!??...")
    assert tokens == [], f"Esperado [], obtenido {tokens}"
    print("PASS: tokenize con solo puntuacion retorna []")


def test_normalize_remove_articles():
    """Eliminacion de articulos en ingles y espanol."""
    result = TextNormalizer.normalize(
        "The cat and a dog", remove_articles=True, language="en"
    )
    assert result == "cat and dog", f"Esperado 'cat and dog', obtenido '{result}'"

    result_es = TextNormalizer.normalize(
        "El gato y un perro", remove_articles=True, language="es"
    )
    assert result_es == "gato y perro", f"Esperado 'gato y perro', obtenido '{result_es}'"
    print("PASS: normalize con remove_articles (en, es)")


# =================================================================
# SECCION 2: F1 Score
# =================================================================

def test_f1_identical():
    """Respuesta identica al expected: F1 = 1.0."""
    r = ReferenceBasedMetrics.f1_score("the cat sat", "the cat sat")
    assert r.value == 1.0, f"Esperado F1=1.0, obtenido {r.value}"
    assert r.details["precision"] == 1.0
    assert r.details["recall"] == 1.0
    print("PASS: f1_score identico = 1.0")


def test_f1_no_overlap():
    """Cero tokens comunes: F1 = 0.0."""
    r = ReferenceBasedMetrics.f1_score("alpha beta", "gamma delta")
    assert r.value == 0.0, f"Esperado F1=0.0, obtenido {r.value}"
    print("PASS: f1_score sin overlap = 0.0")


def test_f1_partial_overlap():
    """Overlap parcial: precision y recall entre 0 y 1."""
    # generated: "the cat sat on the mat" (6 tokens normalizados: the cat sat on the mat)
    # expected:  "the cat"                (2 tokens normalizados: the cat)
    # common: {the: 1, cat: 1} = 2
    # precision = 2/6, recall = 2/2 = 1.0
    # F1 = 2 * (2/6 * 1) / (2/6 + 1) = 2 * 0.3333 / 1.3333 = 0.5
    r = ReferenceBasedMetrics.f1_score("the cat sat on the mat", "the cat")
    assert abs(r.value - 0.5) < 0.01, f"Esperado F1~0.5, obtenido {r.value}"
    assert r.details["recall"] == 1.0
    print("PASS: f1_score overlap parcial (precision < 1, recall = 1)")


def test_f1_partial_overlap_inverse():
    """Inverso: generated es subconjunto de expected."""
    # generated: "the cat" (2 tokens)
    # expected:  "the cat sat on the mat" (6 tokens)
    # common: 2, precision = 2/2 = 1.0, recall = 2/6
    # F1 = 2 * (1.0 * 0.3333) / (1.0 + 0.3333) = 0.5
    r = ReferenceBasedMetrics.f1_score("the cat", "the cat sat on the mat")
    assert abs(r.value - 0.5) < 0.01, f"Esperado F1~0.5, obtenido {r.value}"
    assert r.details["precision"] == 1.0
    print("PASS: f1_score overlap parcial inverso (precision = 1, recall < 1)")


def test_f1_empty_inputs():
    """Inputs vacios retornan 0.0."""
    r1 = ReferenceBasedMetrics.f1_score("", "hello")
    assert r1.value == 0.0
    assert r1.details["reason"] == "empty_input"

    r2 = ReferenceBasedMetrics.f1_score("hello", "")
    assert r2.value == 0.0

    r3 = ReferenceBasedMetrics.f1_score("", "")
    assert r3.value == 0.0
    print("PASS: f1_score inputs vacios = 0.0")


def test_f1_case_insensitive():
    """Normalizacion hace match case-insensitive."""
    r = ReferenceBasedMetrics.f1_score("YES", "yes")
    assert r.value == 1.0, f"Esperado 1.0, obtenido {r.value}"
    print("PASS: f1_score case insensitive")


def test_f1_with_dashes():
    """El caso real de HotpotQA: '1969-1974' vs '1969 until 1974'."""
    # '1969-1974' normaliza a tokens ['1969', '1974']
    # '1969 until 1974' normaliza a tokens ['1969', 'until', '1974']
    # common: {1969: 1, 1974: 1} = 2
    # precision = 2/2 = 1.0, recall = 2/3
    # F1 = 2 * (1.0 * 0.6667) / (1.0 + 0.6667) = 0.8
    r = ReferenceBasedMetrics.f1_score("1969-1974", "1969 until 1974")
    assert r.value > 0.7, (
        f"Esperado F1 > 0.7 (dashes separados), obtenido {r.value}"
    )
    assert r.details["precision"] == 1.0
    print("PASS: f1_score con dashes (1969-1974 vs '1969 until 1974')")


def test_f1_duplicate_tokens():
    """Tokens duplicados: Counter intersection respeta frecuencias."""
    # generated: "the the the" -> Counter({the: 3})
    # expected:  "the the"     -> Counter({the: 2})
    # common: min(3, 2) = 2
    # precision = 2/3, recall = 2/2 = 1.0
    # F1 = 2 * (2/3 * 1.0) / (2/3 + 1.0) = 0.8
    r = ReferenceBasedMetrics.f1_score("the the the", "the the")
    assert abs(r.value - 0.8) < 0.01, f"Esperado F1~0.8, obtenido {r.value}"
    assert r.details["num_common_tokens"] == 2
    print("PASS: f1_score con tokens duplicados respeta frecuencias")


def test_f1_single_token():
    """Un solo token: F1 es 0 o 1 (equivalente a EM)."""
    r_match = ReferenceBasedMetrics.f1_score("yes", "yes")
    assert r_match.value == 1.0

    r_no_match = ReferenceBasedMetrics.f1_score("yes", "no")
    assert r_no_match.value == 0.0
    print("PASS: f1_score single token (equivalente a EM)")


def test_f1_with_accents():
    """Acentos se eliminan en normalizacion: 'café' == 'cafe'."""
    r = ReferenceBasedMetrics.f1_score("café latte", "cafe latte")
    assert r.value == 1.0, f"Esperado 1.0 (acentos normalizados), obtenido {r.value}"
    print("PASS: f1_score con acentos normalizados")


# =================================================================
# SECCION 3: Exact Match
# =================================================================

def test_em_identical():
    r = ReferenceBasedMetrics.exact_match("hello world", "hello world")
    assert r.value == 1.0
    assert r.details["is_match"] is True
    print("PASS: exact_match identico = 1.0")


def test_em_different():
    r = ReferenceBasedMetrics.exact_match("hello", "world")
    assert r.value == 0.0
    assert r.details["is_match"] is False
    print("PASS: exact_match diferente = 0.0")


def test_em_case_insensitive():
    r = ReferenceBasedMetrics.exact_match("YES", "yes")
    assert r.value == 1.0
    print("PASS: exact_match case insensitive")


def test_em_with_punctuation():
    """Puntuacion se elimina: 'hello!' == 'hello'."""
    r = ReferenceBasedMetrics.exact_match("hello!", "hello")
    assert r.value == 1.0, f"Esperado 1.0 (puntuacion eliminada), obtenido {r.value}"
    print("PASS: exact_match con puntuacion")


def test_em_with_accents():
    """Acentos eliminados: 'café' == 'cafe'."""
    r = ReferenceBasedMetrics.exact_match("café", "cafe")
    assert r.value == 1.0, f"Esperado 1.0, obtenido {r.value}"
    print("PASS: exact_match con acentos")


def test_em_with_dashes():
    """Dashes se convierten a espacios: '1969-1974' == '1969 1974'."""
    r = ReferenceBasedMetrics.exact_match("1969-1974", "1969 1974")
    assert r.value == 1.0, f"Esperado 1.0, obtenido {r.value}"
    print("PASS: exact_match con dashes")


def test_em_empty_inputs():
    r1 = ReferenceBasedMetrics.exact_match("", "hello")
    assert r1.value == 0.0
    assert r1.details["reason"] == "empty_input"

    r2 = ReferenceBasedMetrics.exact_match("hello", "")
    assert r2.value == 0.0
    print("PASS: exact_match inputs vacios = 0.0")


def test_em_extra_spaces():
    """Espacios multiples se normalizan."""
    r = ReferenceBasedMetrics.exact_match("  hello   world  ", "hello world")
    assert r.value == 1.0
    print("PASS: exact_match normaliza espacios multiples")


def test_em_without_normalize():
    """Sin normalizacion, case y puntuacion importan."""
    r = ReferenceBasedMetrics.exact_match("Hello!", "hello", normalize=False)
    assert r.value == 0.0
    print("PASS: exact_match sin normalize respeta case y puntuacion")


# =================================================================
# SECCION 4: Accuracy
# =================================================================

def test_accuracy_yes_no_match():
    """Match basico de labels yes/no."""
    r = ReferenceBasedMetrics.accuracy("yes", "yes")
    assert r.value == 1.0
    assert r.details["is_correct"] is True
    print("PASS: accuracy yes/yes = 1.0")


def test_accuracy_yes_no_mismatch():
    r = ReferenceBasedMetrics.accuracy("yes", "no")
    assert r.value == 0.0
    assert r.details["is_correct"] is False
    print("PASS: accuracy yes/no = 0.0")


def test_accuracy_case_insensitive():
    """'YES' == 'yes' con normalizacion."""
    r = ReferenceBasedMetrics.accuracy("YES", "yes")
    assert r.value == 1.0
    print("PASS: accuracy case insensitive")


def test_accuracy_with_extra_text():
    """'Yes, the answer is correct' != 'yes' (EM estricto)."""
    r = ReferenceBasedMetrics.accuracy("Yes, the answer is correct", "yes")
    assert r.value == 0.0, (
        f"Esperado 0.0 (accuracy es EM, no contains), obtenido {r.value}"
    )
    print("PASS: accuracy con texto extra = 0.0 (es EM, no contains)")


def test_accuracy_valid_labels():
    """valid_labels verifica que el label generado es valido."""
    r = ReferenceBasedMetrics.accuracy(
        "yes", "yes", valid_labels=["yes", "no"]
    )
    assert r.value == 1.0
    assert r.details["is_valid_label"] is True

    # Label invalido pero coincide con expected (raro, pero posible)
    r2 = ReferenceBasedMetrics.accuracy(
        "maybe", "maybe", valid_labels=["yes", "no"]
    )
    assert r2.value == 1.0  # coincide, aunque no es label valido
    assert r2.details["is_valid_label"] is False
    print("PASS: accuracy con valid_labels")


def test_accuracy_empty_inputs():
    r = ReferenceBasedMetrics.accuracy("", "yes")
    assert r.value == 0.0
    assert r.details["reason"] == "empty_input"
    print("PASS: accuracy input vacio = 0.0")


def test_accuracy_with_punctuation():
    """'Yes.' normaliza a 'yes', match con 'yes'."""
    r = ReferenceBasedMetrics.accuracy("Yes.", "yes")
    assert r.value == 1.0
    print("PASS: accuracy con puntuacion normalizada")


# =================================================================
# RUNNER
# =================================================================

if __name__ == "__main__":
    # TextNormalizer
    test_normalize_basic()
    test_normalize_empty()
    test_remove_accents()
    test_dashes_to_spaces()
    test_normalize_with_dashes_and_punctuation()
    test_tokenize_punctuation_only()
    test_normalize_remove_articles()

    # F1 Score
    test_f1_identical()
    test_f1_no_overlap()
    test_f1_partial_overlap()
    test_f1_partial_overlap_inverse()
    test_f1_empty_inputs()
    test_f1_case_insensitive()
    test_f1_with_dashes()
    test_f1_duplicate_tokens()
    test_f1_single_token()
    test_f1_with_accents()

    # Exact Match
    test_em_identical()
    test_em_different()
    test_em_case_insensitive()
    test_em_with_punctuation()
    test_em_with_accents()
    test_em_with_dashes()
    test_em_empty_inputs()
    test_em_extra_spaces()
    test_em_without_normalize()

    # Accuracy
    test_accuracy_yes_no_match()
    test_accuracy_yes_no_mismatch()
    test_accuracy_case_insensitive()
    test_accuracy_with_extra_text()
    test_accuracy_valid_labels()
    test_accuracy_empty_inputs()
    test_accuracy_with_punctuation()
