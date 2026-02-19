"""
Tests para DT-9: _extract_score_fallback() regex fix.

Verifica que:
  - Decimales validos 0.X, 1.0 se extraen correctamente
  - Numeros parciales de valores mayores NO se capturan (ej: "10.5" no produce 1.0)
  - Enteros en escala 1-10 con prefijo "score:" se normalizan
  - Fracciones N/M se normalizan
  - Texto sin numeros retorna default 0.5
  - Respuestas reales de LLMs se manejan correctamente

Sin dependencias externas.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.metrics import LLMJudgeMetrics


# =================================================================
# Decimales validos (rango 0-1)
# =================================================================

def test_decimal_zero():
    assert LLMJudgeMetrics._extract_score_fallback("score: 0") == 0.0
    print("PASS: 'score: 0' -> 0.0")


def test_decimal_one():
    assert LLMJudgeMetrics._extract_score_fallback("score: 1") == 1.0
    print("PASS: 'score: 1' -> 1.0")


def test_decimal_fraction():
    r = LLMJudgeMetrics._extract_score_fallback("the score is 0.85")
    assert abs(r - 0.85) < 0.001, f"Esperado 0.85, obtenido {r}"
    print("PASS: 'the score is 0.85' -> 0.85")


def test_decimal_one_point_zero():
    assert LLMJudgeMetrics._extract_score_fallback("score: 1.0") == 1.0
    print("PASS: 'score: 1.0' -> 1.0")


def test_decimal_zero_point_zero():
    assert LLMJudgeMetrics._extract_score_fallback("0.0") == 0.0
    print("PASS: '0.0' -> 0.0")


def test_decimal_bare_in_text():
    """Decimal sin prefijo 'score:' en medio de texto."""
    r = LLMJudgeMetrics._extract_score_fallback("I would rate this 0.7 overall")
    assert abs(r - 0.7) < 0.001, f"Esperado 0.7, obtenido {r}"
    print("PASS: '...0.7 overall' -> 0.7")


# =================================================================
# FALSOS POSITIVOS: numeros > 1 que NO deben capturarse como score
# =================================================================

def test_ten_not_captured_as_one():
    """BUG ORIGINAL: '10.5 points' capturaba '10' -> 1.0. Ahora debe dar default o fraccion."""
    r = LLMJudgeMetrics._extract_score_fallback("10.5 points out of 10")
    # No debe ser 1.0 (el bug original).
    # La fraccion "10" no deberia matchear como decimal.
    # Puede matchear la fraccion implicitamente o dar default.
    assert r != 1.0, f"BUG: '10.5 points out of 10' produjo 1.0 (falso positivo)"
    print(f"PASS: '10.5 points out of 10' -> {r} (no 1.0)")


def test_hundred_not_captured():
    """'100' no debe capturar parcial '1' -> 1.0."""
    r = LLMJudgeMetrics._extract_score_fallback("there are 100 reasons")
    # No hay patron de score, ni fraccion. Default 0.5 o captura de "0" (que seria correcto).
    assert r != 1.0, f"BUG: '100' produjo 1.0"
    print(f"PASS: '...100 reasons' -> {r} (no 1.0)")


def test_twenty_not_captured():
    """'20' no debe producir score espurio."""
    r = LLMJudgeMetrics._extract_score_fallback("about 20 tokens were used")
    assert r != 1.0, f"BUG: '20' produjo 1.0"
    print(f"PASS: '...20 tokens...' -> {r} (no 1.0)")


# =================================================================
# Escala 1-10 con prefijo "score:"
# =================================================================

def test_score_eight_of_ten():
    """'score: 8' en escala 1-10 se normaliza a 0.8."""
    r = LLMJudgeMetrics._extract_score_fallback("score: 8")
    assert abs(r - 0.8) < 0.001, f"Esperado 0.8, obtenido {r}"
    print("PASS: 'score: 8' -> 0.8")


def test_score_three():
    r = LLMJudgeMetrics._extract_score_fallback("score: 3")
    assert abs(r - 0.3) < 0.001, f"Esperado 0.3, obtenido {r}"
    print("PASS: 'score: 3' -> 0.3")


def test_score_ten():
    r = LLMJudgeMetrics._extract_score_fallback("score: 10")
    assert abs(r - 1.0) < 0.001, f"Esperado 1.0, obtenido {r}"
    print("PASS: 'score: 10' -> 1.0")


def test_no_score_prefix_integer_not_normalized():
    """Entero sin prefijo 'score:' no se normaliza (evitar falsos positivos)."""
    # "I give it 8" - sin "score:" no deberia activar escala 1-10
    r = LLMJudgeMetrics._extract_score_fallback("I give it 8")
    # Deberia caer al default 0.5 (no hay decimal 0-1, ni "score:" prefijo, ni fraccion)
    assert r == 0.5, f"Esperado 0.5 (default), obtenido {r}"
    print("PASS: 'I give it 8' (sin score:) -> 0.5 (default)")


# =================================================================
# Fracciones
# =================================================================

def test_fraction_eight_of_ten():
    r = LLMJudgeMetrics._extract_score_fallback("I would rate this 8/10")
    assert abs(r - 0.8) < 0.001, f"Esperado 0.8, obtenido {r}"
    print("PASS: '8/10' -> 0.8")


def test_fraction_one_of_two():
    r = LLMJudgeMetrics._extract_score_fallback("1/2")
    assert abs(r - 0.5) < 0.001, f"Esperado 0.5, obtenido {r}"
    print("PASS: '1/2' -> 0.5")


def test_fraction_ten_of_ten():
    r = LLMJudgeMetrics._extract_score_fallback("10/10")
    assert abs(r - 1.0) < 0.001, f"Esperado 1.0, obtenido {r}"
    print("PASS: '10/10' -> 1.0")


def test_fraction_zero_denominator():
    """Division por 0: fraccion se descarta, decimal captura '0' -> 0.0."""
    r = LLMJudgeMetrics._extract_score_fallback("0/0")
    # Fraccion descartada (denom=0). Decimal captura "0" -> 0.0
    assert r == 0.0, f"Esperado 0.0, obtenido {r}"
    print("PASS: '0/0' -> 0.0 (fraccion descartada, decimal captura 0)")


# =================================================================
# Default y edge cases
# =================================================================

def test_no_numbers_returns_default():
    r = LLMJudgeMetrics._extract_score_fallback("I cannot provide a score")
    assert r == 0.5, f"Esperado 0.5, obtenido {r}"
    print("PASS: texto sin numeros -> 0.5 (default)")


def test_empty_string():
    r = LLMJudgeMetrics._extract_score_fallback("")
    assert r == 0.5, f"Esperado 0.5, obtenido {r}"
    print("PASS: string vacio -> 0.5 (default)")


def test_realistic_llm_response_json_fail():
    """Respuesta tipica de LLM cuando no retorna JSON: 'Based on my analysis, the faithfulness score is 0.75'."""
    r = LLMJudgeMetrics._extract_score_fallback(
        "Based on my analysis, the faithfulness score is 0.75"
    )
    assert abs(r - 0.75) < 0.001, f"Esperado 0.75, obtenido {r}"
    print("PASS: respuesta tipica LLM con decimal -> 0.75")


def test_realistic_llm_verbose():
    """'The response is well-grounded. Score: 0.9'"""
    r = LLMJudgeMetrics._extract_score_fallback(
        "The response is well-grounded. Score: 0.9"
    )
    assert abs(r - 0.9) < 0.001, f"Esperado 0.9, obtenido {r}"
    print("PASS: respuesta LLM verbose con 'Score: 0.9' -> 0.9")


if __name__ == "__main__":
    # Decimales validos
    test_decimal_zero()
    test_decimal_one()
    test_decimal_fraction()
    test_decimal_one_point_zero()
    test_decimal_zero_point_zero()
    test_decimal_bare_in_text()

    # Falsos positivos
    test_ten_not_captured_as_one()
    test_hundred_not_captured()
    test_twenty_not_captured()

    # Escala 1-10
    test_score_eight_of_ten()
    test_score_three()
    test_score_ten()
    test_no_score_prefix_integer_not_normalized()

    # Fracciones
    test_fraction_eight_of_ten()
    test_fraction_one_of_two()
    test_fraction_ten_of_ten()
    test_fraction_zero_denominator()

    # Default y edge cases
    test_no_numbers_returns_default()
    test_empty_string()
    test_realistic_llm_response_json_fail()
    test_realistic_llm_verbose()
