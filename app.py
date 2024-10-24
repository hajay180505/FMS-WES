import streamlit as st
import time

st.title("Dynamic Overwriting Example")

# Create a placeholder to overwrite content
placeholder = st.empty()

# Loop to update the same placeholder
for i in range(1, 11):
    with placeholder.container():
        st.write(f"Overwriting... Step {i}/10")  # Overwrites the previous content
        st.progress(i * 10)  # Progress bar inside the same placeholder
    time.sleep(0.5)  # Simulate some delay

# After the loop, show final message
placeholder.write("Completed! Final content displayed.")
