import json

from physicalrsi.cli import main


def test_mainline_help_describes_the_core_commands(capsys):
    assert main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "evolve" in output
    assert "harness" in output
    assert "skills" in output
    assert "compatibility surface" in output


def test_cli_designs_and_reads_a_user_harness(tmp_path, capsys):
    definition = tmp_path / "harness-definition.json"
    definition.write_text(json.dumps({
        "name": "user-agent",
        "mutable": ["skills"],
        "evaluation": {"split": "development"},
    }))
    assert main([
        "harness", "design", "--workspace", str(tmp_path),
        "--definition", str(definition),
    ]) == 0
    designed = json.loads(capsys.readouterr().out)
    assert designed["schema"] == "physicalrsi.harness/v1"
    assert designed["name"] == "user-agent"

    assert main(["harness", "show", "--workspace", str(tmp_path)]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["sha256"] == designed["sha256"]
