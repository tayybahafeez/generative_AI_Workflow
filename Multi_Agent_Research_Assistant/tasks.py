from crewai import Task
from agents import search_agent, summary_agent, critic_agent, idea_agent

def get_tasks(topic):
    return [
        Task(
            description=f"Search for the most recent research papers on {topic}",
            expected_output="List of 5 papers with URLs and summaries",
            agent=search_agent
        ),
        Task(
            description="Summarize the papers concisely with technical highlights",
            expected_output="Summaries with key contributions",
            agent=summary_agent
        ),
        Task(
            description="Critically review each paper's methodology and claims",
            expected_output="Critical evaluation report",
            agent=critic_agent
        ),
        Task(
            description="Generate new research ideas based on the paper findings",
            expected_output="3 novel research ideas with justification",
            agent=idea_agent
        )
    ]
