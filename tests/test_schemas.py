from pathlib import Path

import pytest
from pydantic import ValidationError

from mini_decision.schemas import Example, GeneratedPair, Judgment, Kind, read_jsonl, write_jsonl

VALID = {
    "pair_id": "p_00123",
    "context": "Le client a réglé sa facture le 3 mars.",
    "desc_true": "La facture est payée, le client ne doit plus rien.",
    "desc_false": "Le paiement est manquant ou en retard, il reste un solde dû.",
    "label": 0.95,
    "kind": "true",
    "round": 0,
}


def test_example_valid() -> None:
    ex = Example.model_validate(VALID)
    assert ex.kind is Kind.TRUE


@pytest.mark.parametrize("label", [-0.1, 1.2])
def test_label_out_of_range(label: float) -> None:
    with pytest.raises(ValidationError):
        Example.model_validate({**VALID, "label": label})


def test_off_topic_must_be_half() -> None:
    with pytest.raises(ValidationError):
        Example.model_validate({**VALID, "kind": "off_topic", "label": 0.9})
    Example.model_validate({**VALID, "kind": "off_topic", "label": 0.5})


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError):
        Example.model_validate({**VALID, "extra": 1})


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    items = [Example.model_validate(VALID), Example.model_validate({**VALID, "pair_id": "p_2"})]
    path = tmp_path / "x.jsonl"
    assert write_jsonl(path, items) == 2
    assert read_jsonl(path, Example) == items


def test_llm_output_schemas() -> None:
    pair = GeneratedPair.model_validate(
        {
            "desc_true": "a",
            "desc_false": "b",
            "contexts": [{"text": "c", "kind": "counterfactual", "source_index": 0}],
        }
    )
    assert pair.contexts[0].kind is Kind.COUNTERFACTUAL
    with pytest.raises(ValidationError):
        Judgment.model_validate({"off_topic": False, "p_true": 1.5})
