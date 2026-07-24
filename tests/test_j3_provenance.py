import hashlib
import json
from pathlib import Path

from scripts.correct_j3_provenance import correct_j3_provenance
from speech_jscc.diagnostics.j3_narrowband import j3_initialization_metadata


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_correction_preserves_original_metrics_and_checkpoint(tmp_path):
    j2 = tmp_path / "j2"; j3 = tmp_path / "j3"; j2.mkdir(); j3.mkdir()
    (j2 / "summary.json").write_text(json.dumps({"classification":"PASS","provenance":{"stage_name":"j2_strong_barrage","model_architecture":"conv_conformer_v1"}}))
    (j2 / "diagnostic_last.pt").write_bytes(b"j2 checkpoint")
    original = {"classification":"PASS","steps":4096,"validation":{"loss":.7},"provenance":{"initialization_mode":"fresh_initialization_control","parent_checkpoint":str(j2 / "diagnostic_last.pt"),"architecture_version":"conv_conformer_v1","preprocessing":{"codec_type":"speechtokenizer"},"git_commit":"abc"}}
    (j3 / "summary.json").write_text(json.dumps(original)); (j3 / "diagnostic_last.pt").write_bytes(b"j3 checkpoint")
    checkpoint_hash = sha(j3 / "diagnostic_last.pt")
    accepted = correct_j3_provenance(j3 / "summary.json", j3 / "diagnostic_last.pt", j2 / "summary.json", j2 / "diagnostic_last.pt", config_hash="cfg", initial_weights_loaded=True)
    assert json.loads((j3 / "summary.json").read_text()) == original
    assert sha(j3 / "diagnostic_last.pt") == checkpoint_hash
    corrected = json.loads((j3 / "summary.corrected.json").read_text())
    assert corrected["validation"] == original["validation"]
    assert corrected["provenance"]["initialization_mode"] == "j2_transfer"
    assert corrected["provenance"]["initial_weights_loaded"] is True
    assert accepted["classification"] == "ACCEPTED_PASS"


def test_future_j3_transfer_metadata_never_reports_fresh():
    value = j3_initialization_metadata("parent.pt", "abc", "summary.json", "def")
    assert value["initialization_mode"] == "j2_transfer"
    assert value["initialization_source_stage"] == "j2_strong_barrage_boundary"
    assert value["initial_weights_loaded"] is True
    assert "fresh_initialization_control" not in value.values()
