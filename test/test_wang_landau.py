from phaseedge.orchestration.jobs.wang_landau import WangLandauSpec, run_wang_landau
from scripts import wang_landau as wl_cli


def _dummy_ce() -> dict:
    return {"sampling": {"counts": {"A": 1, "B": 1}}}


def test_run_wang_landau_deterministic() -> None:
    ce = _dummy_ce()
    spec = WangLandauSpec(
        ce_key="dummy",
        composition={"A": 0.5, "B": 0.5},
        steps=500,
        bin_size=0.1,
        n_samples=5,
        seed=123,
    )
    res1 = run_wang_landau(spec, ce)
    res2 = run_wang_landau(spec, ce)
    assert res1 == res2
    assert len(res1["dos"]) == 10
    assert len(res1["samples"]) == 5
    assert res1["run_key"] == spec.run_key()


def test_cli_wl_run(monkeypatch, capsys) -> None:
    monkeypatch.setattr(wl_cli, "lookup_ce_by_key", lambda key: _dummy_ce())
    argv = [
        "--ce-key",
        "dummy",
        "--composition",
        "A:0.5,B:0.5",
        "--steps",
        "100",
        "--bin-size",
        "0.1",
        "--n-samples",
        "2",
        "--seed",
        "7",
    ]
    wl_cli.main(argv)
    out = capsys.readouterr().out
    assert "run_key:" in out
    assert "dos_bins: 10" in out
    assert "n_samples: 2" in out
