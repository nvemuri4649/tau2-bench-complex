# Copyright Sierra
from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.tools import AirlineTools
from tau2.domains.airline.utils import (
    AIRLINE_DB_PATH,
    AIRLINE_POLICY_PATH,
    AIRLINE_TASK_SET_PATH,
    AIRLINE_COMPLEX_TASK_SET_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file


def get_environment(
    db: Optional[FlightDB] = None,
    solo_mode: bool = False,
) -> Environment:
    if solo_mode:
        raise ValueError("Airline domain does not support solo mode")
    if db is None:
        db = FlightDB.load(AIRLINE_DB_PATH)
    tools = AirlineTools(db)
    with open(AIRLINE_POLICY_PATH, "r") as fp:
        policy = fp.read()
    return Environment(
        domain_name="airline",
        policy=policy,
        tools=tools,
    )


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    """Get tasks for the airline domain.
    
    Args:
        task_split_name: The name of the task split to load.
            - "base", "train", "test": Standard task splits from tasks.json
            - "complex": Load complex tasks from complex_tasks.json
            - None: Load all tasks from tasks.json
    
    Returns:
        List of Task objects.
    """
    # Handle complex tasks specially
    if task_split_name == "complex":
        if not AIRLINE_COMPLEX_TASK_SET_PATH.exists():
            raise ValueError(
                f"Complex tasks file not found at {AIRLINE_COMPLEX_TASK_SET_PATH}"
            )
        tasks = load_file(AIRLINE_COMPLEX_TASK_SET_PATH)
        return [Task.model_validate(task) for task in tasks]
    
    # Standard task loading
    tasks = load_file(AIRLINE_TASK_SET_PATH)
    tasks = [Task.model_validate(task) for task in tasks]
    if task_split_name is None:
        return tasks
    task_splits = get_tasks_split()
    if task_split_name not in task_splits:
        raise ValueError(
            f"Invalid task split name: {task_split_name}. Valid splits are: {list(task_splits.keys())} or 'complex'"
        )
    return [task for task in tasks if task.id in task_splits[task_split_name]]


def get_tasks_split() -> dict[str, list[str]]:
    split_file = (
        Path(AIRLINE_TASK_SET_PATH).parent
        / f"split_{Path(AIRLINE_TASK_SET_PATH).stem}.json"
    )
    return load_file(split_file)
