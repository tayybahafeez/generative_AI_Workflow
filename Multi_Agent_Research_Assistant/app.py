import streamlit as st
from crew_runner import run_research
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

st.title("🧠 Multi-Agent AI Research Assistant")

topic = st.text_input("Enter your research topic (e.g., LLMs in healthcare)")

if st.button("Run Research Crew"):
    if topic:
        status_placeholder = st.empty()  # Placeholder for dynamic status updates
        output_placeholder = st.empty()  # Placeholder for final result
        logs = []

        try:
            def progress_callback(message):
                logs.append(message)
                status_placeholder.markdown(f"🧠 `{message}`")

            # Run research with step-by-step logging
            output = run_research(topic, progress_callback=progress_callback)

            st.success("Done!")
            output_placeholder.text_area("📝 Research Output", output, height=400)
        except Exception as e:
            logging.error("Error during crew execution", exc_info=True)
            st.error(f"Something went wrong: {e}")
    else:
        st.warning("Please enter a topic.")
