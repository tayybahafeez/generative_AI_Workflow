def run_research(topic, progress_callback=None):
    from crewai.process import Process
    from tasks import get_tasks
    from deepseek_llm import call_deepseek
    from crewai import Crew, LLM

    LLM.call = lambda self, prompt, *args, **kwargs: call_deepseek(prompt)

    tasks = get_tasks(topic)
    
    # Add progress logs before kickoff
    if progress_callback:
        for task in tasks:
            progress_callback(f"Assigned task: {task.description} → {task.agent.role}")

    crew = Crew(
        agents=[t.agent for t in tasks],
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()

    if progress_callback:
        progress_callback("✅ All tasks completed.")

    return result
