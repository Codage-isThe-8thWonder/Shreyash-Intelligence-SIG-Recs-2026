import json
from datetime import datetime
from pathlib import Path


FINALE_DIR = (
    Path(__file__).resolve().parent.parent
)

DEFAULT_OUTPUT_DIR = (
    FINALE_DIR / "test_results"
)


class FinaleTraceManager:

    def __init__(
        self,
        output_directory=DEFAULT_OUTPUT_DIR,
    ):

        self.output_directory = Path(
            output_directory
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def create_trace(self, request):

        return {
            "trace_id": datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            ),

            "started_at": datetime.now().isoformat(),

            "input": request,

            "route": [],

            "steps": [],

            "final_output": None,

            "status": "running",
        }

    def add_step(
        self,
        trace,
        agent,
        input_data,
        output=None,
        status="running",
        error=None,
    ):

        step = {
            "step": len(
                trace["steps"]
            ) + 1,

            "agent": agent,

            "input": input_data,

            "status": status,
        }

        if output is not None:
            step["output"] = output

        if error is not None:
            step["error"] = error

        trace["steps"].append(
            step
        )

    def save(
        self,
        trace,
        filename=None,
    ):

        if filename is None:

            filename = (
                f"{trace['trace_id']}.json"
            )

        output_path = (
            self.output_directory
            / filename
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                trace,
                file,
                indent=4,
                ensure_ascii=False,
                default=str,
            )

        return output_path