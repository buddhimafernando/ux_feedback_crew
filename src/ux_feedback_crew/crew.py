import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from .tools import (
    analyze_ui_screenshot,
    evaluate_heuristics,
    generate_feedback,
    create_wireframe
)


@CrewBase
class UxFeedbackCrew():
    """UX Feedback Crew - Multi-agent system for UI evaluation"""
    
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    #  Define the Gemini LLM for the agents to use
    def __init__(self):
        self.gemini_llm = LLM(
            model="gemini/gemini-2.5-flash", # Or gemini-2.0-flash-exp
            api_key=os.getenv("GEMINI_API_KEY")
        )

        self.gemini_preview_llm = LLM(
            model="gemini/gemini-3-flash-preview", 
            api_key=os.getenv("GEMINI_API_KEY")
        )

    @agent
    def vision_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['vision_analyst'],
            tools=[analyze_ui_screenshot],
            llm=self.gemini_llm, 
            verbose=True,   
            allow_delegation=False # Prevent it from asking other agents for help
        )
    
    @agent
    def heuristic_evaluator(self) -> Agent:
        return Agent(
            config=self.agents_config['heuristic_evaluator'],
            tools=[evaluate_heuristics],
            llm=self.gemini_llm, # 2. Assign the LLM here
            verbose=True
        )
    
    @agent
    def feedback_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['feedback_specialist'],
            llm=self.gemini_llm, # 2. Assign the LLM here
            tools=[generate_feedback],
            verbose=True
        )
    
    @agent
    def wireframe_designer(self) -> Agent:
        return Agent(
            config=self.agents_config['wireframe_designer'],
            llm=self.gemini_preview_llm, # 2. Assign the LLM here
            tools=[create_wireframe],
            verbose=True
        )
    
    # Task names MUST match the YAML keys
    @task
    def analyze_ui(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_ui'],
        )
    
    @task
    def evaluate_heuristics(self) -> Task:
        return Task(
            config=self.tasks_config['evaluate_heuristics'],
        )
    
    @task
    def generate_feedback(self) -> Task:
        return Task(
            config=self.tasks_config['generate_feedback'],
        )
    
    @task
    def create_wireframe(self) -> Task:
        return Task(
            config=self.tasks_config['create_wireframe'],
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the UX Feedback Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            max_iter=1   
        )