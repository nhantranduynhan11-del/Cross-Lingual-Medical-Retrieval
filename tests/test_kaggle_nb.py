import sys

import pytest
import yaml

from mir import config, kaggle_nb


def test_locate(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle_nb, "INPUT", tmp_path)
    for d in ("mir-code/mir", "nested/mir-data/Data", "mir-emb-m0/emb/structure", "mir-emb-m1/emb/structure"):
        (tmp_path / d).mkdir(parents=True)
    (tmp_path / "mir-code/mir/__init__.py").write_text("")
    (tmp_path / "nested/mir-data/Data/query.parquet").write_text("")
    assert kaggle_nb.locate("mir/__init__.py") == tmp_path / "mir-code"
    assert kaggle_nb.locate("Data/query.parquet") == tmp_path / "nested" / "mir-data"      # dataset lồng thư mục
    assert kaggle_nb.locate_all("emb/structure") == [tmp_path / "mir-emb-m0", tmp_path / "mir-emb-m1"]
    with pytest.raises(FileNotFoundError, match="Add Input"):
        kaggle_nb.locate("bm25/stats.json")


def test_write_config_overrides(tmp_path):
    dst = kaggle_nb.write_config(config.ROOT / "configs/baseline.yaml", tmp_path / "c.yaml",
                                 {"encode.shard_size": 2500, "retrieve.queries": "sample/query.parquet"})
    c = yaml.safe_load(dst.read_text(encoding="utf-8"))
    assert c["encode"]["shard_size"] == 2500 and c["retrieve"]["queries"] == "sample/query.parquet"
    assert c["clean"]["force_drop"]["thanhnien.vn"] == ["tin liên quan"]                    # phần khác giữ nguyên
    with pytest.raises(KeyError, match="encode.shard_sise"):
        kaggle_nb.write_config(config.ROOT / "configs/baseline.yaml", tmp_path / "c.yaml", {"encode.shard_sise": 1})


def test_sh_and_par(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(kaggle_nb, "WORK", tmp_path)
    assert kaggle_nb._py("python -m mir.x --a 1").startswith(f'"{sys.executable}" -m mir.x')
    assert kaggle_nb._py("pythonx -m y") == "pythonx -m y"
    assert kaggle_nb.sh('python -c "print(\'xin chào\')"', step="a") == 0
    assert "xin chào" in capsys.readouterr().out and "a" in kaggle_nb.T
    with pytest.raises(RuntimeError, match="LỖI"):
        kaggle_nb.sh('python -c "import sys; sys.exit(3)"')
    kaggle_nb.par(['python -c "print(1)"', 'python -c "print(2)"'], "hai lệnh")
    assert (tmp_path / "log_hai_l_nh_1.txt").read_text().strip() == "2"
    with pytest.raises(RuntimeError, match="LỖI ở"):
        kaggle_nb.par(['python -c "print(1)"', 'python -c "import sys; sys.exit(1)"'], "lỗi")
