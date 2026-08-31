import argparse
import uuid

from typing import List, TypedDict

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

# loads environement variables from .env file
load_dotenv(override=True)

class Task(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Unique Identifier for the task.")
    task_name: str = Field(description="Name of the Task")
    task_description: str = Field(description="Description of the task.")
    estimated_day: int = Field(description="Estimated number of days to complete the task.")

class TaskList(BaseModel):
    tasks: List[Task] = Field(description="List of Tasks")


class TaskDependency(BaseModel):
    task: Task = Field(description="Task")
    dependency_tasks: List[Task] = Field("List of Dependent task")


class TeamMember(BaseModel):
    name: str = Field(description="Name of the Team Member")
    profile: str = Field(description="Profile of the Team Member")

class Team(BaseModel):
    team_members: List[TeamMember] = Field(description="List of Team Members")

class TaskAllocation(BaseModel):
    task: Task = Field(description="Task")
    team_member: TeamMember = Field(description="Team member assigned to the task.")

class TaskSchedule(BaseModel):
    task: Task = Field(description="Task")
    start_day: int = Field(description="Start day of the task")
    end_day: int = Field(description="End day of the task")

class DependencyList(BaseModel):
    dependencies: List[TaskDependency] = Field(description="List of task dependencies")

class Schedule(BaseModel):
    schedule: List[TaskSchedule] = Field(description="List of task schedules")

class TaskAllocationList(BaseModel):
    task_allocations: List[TaskAllocation] = Field(description="List of task allocations")

class Risk(BaseModel):
    task: Task = Field(description="Task")
    score: int = Field(description="Risk score associated with the task from 0 to 10")

class RiskList(BaseModel):
    risks: List[Risk] = Field(description="List of risks")

class AgenState(TypedDict):
    project_description: str
    team: Team
    tasks: TaskList
    dependencies: DependencyList
    schedule: Schedule
    task_allocations: TaskAllocationList
    risks: RiskList
    project_risk_score: int
    iteration_number: int
    max_iteration: int
    insights: int
    schedule_iteration: List[Schedule]
    task_allocations_iteration: List[TaskAllocationList]
    risks_iteration: List[RiskList]
    project_risk_score_iteration: List[int]

def load_llm(model_provider: str):
    if model_provider.lower == "Azure":
        return AzureChatOpenAI(deployment_name="gpt-4o-mini")
    if model_provider.lower == "openai":
        return ChatOpenAI(model="gpt-4o-mini")
    raise ValueError("model_provider must be either 'Azure' or 'OpenAi'")

def tas_generation_node(state: AgenState, llm):
    prompt = f"""
    You are an expert project manager tasked with analyzing the following project description.
    {state["project_description"]}

    Objectives:
    1) Extract actionable and realistic tasks with estimated duration of days.
    2) If a task takes longer than 5 days, split it into smaller independent tasks.
    3) Keep tasks clerly defined and execution-friendly.
   """
    tasks = llm.with_structured_output(TaskList).invoke(prompt)
    return {"tasks": tasks}

def task_dependency_node(state: AgenState, llm):
    prompt = f"""
    You are a project scheduler mapping dependencies.
    Given Tasks: {state["tasks"]}
    
    For each task:
    - Identify task that depends on it.
    - Keep dependencies realistic and minimal
   """
    dependencies = llm.with_structured_output(DependencyList).invoke(prompt)
    return {"dependencies": dependencies}


def task_scheduler(state: AgenState, llm):
    prompt = f"""
    Create an optimized task schedule.

    Tasks: {state["tasks"]}
    Dependencies: {state["dependencies"]}
    Previous Insights: {state["insights"]}
    Previous Schedules: {state["schedule_iteration"]}

    Rules:
    - Respect dependencies.
    - Parallelize where possible.
    - Reduce total duration without harming feasibility.
    """
    schedule = llm.with_structured_output(Schedule).invoke(prompt)
    schedule_iteration = list(state["schedule_iteration"])
    schedule_iteration.append(schedule)
    return {"schedule": schedule, "schedule_iteration": schedule_iteration}


    