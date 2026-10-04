import json
from datetime import datetime
from pathlib import Path


PART_C_DIR = Path(__file__).resolve().parent.parent

DEFAULT_TRACE_DIR = (
    PART_C_DIR / "traces"
)


class TraceManager:
    """
    Creates and persists complete pipeline traces.
    """

    def __init__(
        self,
        trace_directory=DEFAULT_TRACE_DIR,
    ):

        self.trace_directory = Path(
            trace_directory
        )

        self.trace_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def create_trace(self, request):
        """
        Create a new trace object.
        """

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

    def save(self, trace):
        """
        Save trace as JSON.
        """

        output_file = (
            self.trace_directory
            / f"{trace['trace_id']}.json"
        )

        with open(
            output_file,
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

        return output_file

    def add_step(
        self,
        trace,
        agent,
        input_data,
        output=None,
        status="running",
        step_number=None,
        attempt=None,
        error=None,
    ):
        """
        Add one agent execution step to the trace.
        """

        if step_number is None:
            step_number = (
                len(trace["steps"]) + 1
            )

        step = {
            "step": step_number,
            "agent": agent,
            "input": input_data,
            "status": status,
        }

        if attempt is not None:
            step["attempt"] = attempt

        if output is not None:
            step["output"] = output

        if error is not None:
            step["error"] = error

        trace["steps"].append(step)

        return step