from pathlib import Path
from unittest.mock import patch

from engineering_case_studies.executors import LocalExecutor, SlurmExecutor


def test_local_executor_runs_command():
    result = LocalExecutor().run(["python", "-c", "print('ok')"])
    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_slurm_submission_parses_parsable_job_id():
    completed = type("Completed", (), {"stdout": "12345;cluster\n"})()
    executor = SlurmExecutor()
    with patch("subprocess.run", return_value=completed) as run:
        result = executor.submit(
            Path("job.sbatch"),
            exports={"STUDY_CONFIG": "study.toml"},
        )
    assert result.job_id == "12345"
    command = run.call_args.args[0]
    assert command[:2] == ["sbatch", "--parsable"]
    assert "--export=ALL,STUDY_CONFIG=study.toml" in command
