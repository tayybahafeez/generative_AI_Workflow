from crewai import Agent

search_agent = Agent(
    role="Search Expert",
    goal="Find recent AI research papers",
    backstory="Expert in academic research and databases",
    verbose=True
)

summary_agent = Agent(
    role="Paper Summarizer",
    goal="Summarize complex AI papers",
    backstory="Academic researcher who summarizes papers",
    verbose=True
)

critic_agent = Agent(
    role="Critical Reviewer",
    goal="Evaluate flaws in research papers",
    backstory="Reviewer for top AI journals",
    verbose=True
)

idea_agent = Agent(
    role="Innovation Researcher",
    goal="Generate new research ideas",
    backstory="Thinks outside the box to suggest next steps",
    verbose=True
)
