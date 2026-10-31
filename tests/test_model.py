"""Tests de la section 8 : formes, nombre de paramètres, antisymétrie, porte, log-densité QDA."""

import math

import pytest
import torch
from torch.distributions import MultivariateNormal

from mini_decision.config import Config, load_config
from mini_decision.model import Batch, build_model, count_parameters
from mini_decision.model.heads import Cloud, QDAHead, gaussian_score, lower_from_flat
from tests.conftest import ROOT

VOCAB = 4000


def config(*overrides: str) -> Config:
    return load_config(ROOT / "configs" / "base.yaml", list(overrides))


def random_batch(b: int = 5, seed: int = 0) -> Batch:
    g = torch.Generator().manual_seed(seed)

    def field(length: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.randint(6, VOCAB, (b, length), generator=g)
        lengths = torch.randint(2, length + 1, (b,), generator=g)
        mask = torch.arange(length) < lengths[:, None]
        mask[:, 0] = True
        return ids.masked_fill(~mask, 0), mask

    return Batch(*field(24), *field(9), *field(12))


def swapped(batch: Batch) -> Batch:
    return Batch(
        batch.ctx_ids, batch.ctx_mask, batch.false_ids, batch.false_mask, batch.true_ids, batch.true_mask
    )


@pytest.mark.parametrize("encoder", ["ngram", "cnn", "transformer"])
@pytest.mark.parametrize("head", ["qda", "dot"])
def test_forward_shapes_and_gradients(encoder: str, head: str) -> None:
    cfg = config(f"model.encoder={encoder}", f"model.head={head}", "model.cov=full")
    model = build_model(VOCAB, cfg.model)
    model.set_rda_alpha(0.5)
    out = model(random_batch())
    assert out.logit.shape == out.s_true.shape == out.gate.shape == (5,)
    assert out.z.shape == (5, cfg.model.k)
    assert torch.isfinite(out.logit).all()
    out.logit.sum().backward()
    assert model.encoder.token_embedding.weight.grad is not None


def test_transformer_parameter_count() -> None:
    model = build_model(VOCAB, config("model.encoder=transformer").model)
    assert 800_000 <= count_parameters(model) <= 1_500_000


@pytest.mark.parametrize("head", ["qda", "dot"])
@pytest.mark.parametrize("alpha", [0.0, 0.4, 1.0])
def test_swapping_descriptions_flips_probability(head: str, alpha: float) -> None:
    torch.manual_seed(0)
    model = build_model(VOCAB, config(f"model.head={head}", "model.cov=full").model).eval()
    model.set_rda_alpha(alpha)
    assert model.output.bias.item() == 0.0
    with torch.no_grad():
        model.output.tau.fill_(-3.0)
        batch = random_batch()
        p = torch.sigmoid(model(batch).logit)
        p_swapped = torch.sigmoid(model(swapped(batch)).logit)
    assert torch.allclose(p + p_swapped, torch.ones_like(p), atol=1e-5)


@pytest.mark.parametrize("cov", ["diag", "full"])
def test_far_context_gives_one_half(cov: str) -> None:
    torch.manual_seed(0)
    model = build_model(VOCAB, config(f"model.cov={cov}", "model.tau_init=0.0").model).eval()
    k = model.cfg.k
    h = torch.randn(3, model.encoder.d_model)
    z_far = torch.full((3, k), 50.0)  # très loin des deux nuages
    with torch.no_grad():
        s_true, s_false = model.head.score(z_far, h), model.head.score(z_far, -h)
        logit, gate = model.output(s_true, s_false)
    p = torch.sigmoid(logit)
    assert (gate < 1e-3).all()
    assert ((p >= 0.45) & (p <= 0.55)).all()


@pytest.mark.parametrize("alpha", [1.0, 0.3, 0.0])
@pytest.mark.parametrize("full", [False, True])
def test_qda_score_matches_multivariate_normal(alpha: float, full: bool) -> None:
    torch.manual_seed(1)
    k, b = 6, 4
    diag = torch.rand(b, k) + 0.3
    off = 0.3 * torch.randn(b, k * (k - 1) // 2) if full else None
    chol = lower_from_flat(diag, off, k)
    shared_off = 0.2 * torch.randn(k * (k - 1) // 2) if full else None
    shared = lower_from_flat(torch.rand(k) + 0.5, shared_off, k)
    mu, z = torch.randn(b, k), torch.randn(b, k)

    score = gaussian_score(z, Cloud(mu, chol), shared, alpha, diagonal=not full)

    precision = alpha * chol @ chol.transpose(-1, -2) + (1 - alpha) * shared @ shared.T
    expected = MultivariateNormal(mu, precision_matrix=precision).log_prob(z)
    assert torch.allclose(score - k / 2 * math.log(2 * math.pi), expected, atol=1e-4)


def test_qda_precision_is_bounded() -> None:
    head = QDAHead(8, config().model)
    cloud = head.cloud(1000 * torch.randn(16, 8))
    d = torch.diagonal(cloud.chol, dim1=-2, dim2=-1)
    assert (d >= 1 / 5.0 - 1e-6).all() and (d <= 1 / 0.05 + 1e-4).all()


def test_tau_auto_initialised_on_first_training_batch() -> None:
    model = build_model(VOCAB, config("model.tau_init=auto").model).train()
    out = model(random_batch())
    assert bool(model.output.tau_ready)
    assert out.gate.mean().item() == pytest.approx(0.5, abs=0.15)
