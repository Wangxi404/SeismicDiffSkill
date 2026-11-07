"""Exercise figure artefacts and evidence-integrity guards, not plot wording."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("seismic_visualize", Path(__file__).resolve().parents[1]/"skills/differentiate-seismic-solver/scripts/visualize.py")
visualize = importlib.util.module_from_spec(spec); spec.loader.exec_module(visualize)


def panel(key, **extra):
    return dict(kind="map", key=key, label=key, units="m/s", extent=[0,3,2,0],
                xlabel="x (m)", ylabel="z (m)", scale_group="velocity", **extra)


def test_real_exports_share_scale_and_preserve_raw_arrays(tmp_path):
    data={"a":np.arange(6.).reshape(2,3),"b":np.arange(6.,12.).reshape(2,3)}
    originals={k:v.copy() for k,v in data.items()}
    config={"figures":[dict(name="models",title="Native test arrays",caption="All values retained.",columns=2,panels=[panel("a"),panel("b")])]}
    report=visualize.render(data,config,tmp_path)
    stats=[p["raw_statistics"] for p in report["figures"][0]["panels"]]
    assert stats[0]["display_limits"]==stats[1]["display_limits"]==[0.,11.]
    assert all(p["clipped_count"]==0 for p in stats)
    for suffix in ("png","pdf","svg","alignment.json"):
        assert (tmp_path/f"models.{suffix}").stat().st_size>0
    for k in data:np.testing.assert_array_equal(data[k],originals[k])
    assert json.loads((tmp_path/"manifest.json").read_text())["figures"]


def test_clipping_is_counted_and_different_units_cannot_share_scale(tmp_path):
    data={"a":np.array([[-10.,0.,1.,20.]])}
    config={"figures":[dict(name="clipped",title="Declared clipping",caption="Clipped only for display.",panels=[panel("a",symmetric=True,percentile=50)])]}
    report=visualize.render(data,config,tmp_path)
    stats=report["figures"][0]["panels"][0]["raw_statistics"]
    assert stats["clipped_count"]==2 and stats["clipped_fraction"]==.5
    p2=panel("a");p2["units"]="Pa"
    config["figures"][0]["panels"]=[panel("a"),p2]
    with pytest.raises(ValueError,match="shared scale"):
        visualize.render(data,config,tmp_path)


def test_snapshot_indices_and_animation_times_are_not_fabricated(tmp_path):
    data={"u":np.arange(12.).reshape(2,2,3),"t":np.array([.2,.7])}
    animation=panel("u");animation.update(name="native",time_key="t",frames=[0,1],label="Native u",time_units="s",fps=2,caption="Both original frames.")
    report=visualize.render(data,{"animations":[animation]},tmp_path)
    assert (tmp_path/"native.gif").stat().st_size>0
    assert report["animations"][0]["selected_times"]==[.2,.7]
    with pytest.raises(TypeError):visualize.field(data,panel("u",frame=.5))
    with pytest.raises(ValueError):visualize.field(data,panel("u",frame=-1))


def test_invalid_data_and_overwritten_evidence_are_rejected(tmp_path):
    with pytest.raises(ValueError):visualize.array({"x":np.array([np.nan])},"x")
    with pytest.raises(ValueError):visualize.array({"x":np.array([1+2j])},"x")
    item=dict(name="same",title="Test",caption="Test",panels=[panel("a")])
    with pytest.raises(ValueError,match="duplicate"):
        visualize.render({"a":np.ones((2,3))},{"figures":[item,item]},tmp_path)
