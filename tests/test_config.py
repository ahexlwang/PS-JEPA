from psjepa.config import load_config


def test_recursive_config_include(tmp_path):


    (tmp_path / "base.yaml").write_text(
        "model:\n  d_model: 128\n  quantize: true\nloss:\n  code: 0.1\n  s2p: 1.0\n"
    )
    (tmp_path / "child.yaml").write_text(
        "include: base.yaml\nexperiment: child\nmodel:\n  quantize: false\n"
    )
    config = load_config(tmp_path / "child.yaml")
    assert config["model"]["d_model"] == 128          # inherited from the include
    assert config["model"]["quantize"] is False       # overridden by the child
    assert config["loss"]["code"] == 0.1 and config["loss"]["s2p"] == 1.0
    assert config["experiment"] == "child"
