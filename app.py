import streamlit as st
from src import CodeValidator
import json
import torch

def main():
    st.title("Next Token Prediction - Code Testing UI")
    st.write("Enter your Python code and see the predictions and test results")

    # Initialize validator
    if 'validator' not in st.session_state:
        with st.spinner("Loading model... This might take a few minutes..."):
            try:
                st.session_state.validator = CodeValidator(test_mode=True)  # Force test mode for now
                st.success("✅ Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.session_state.validator = CodeValidator(test_mode=True)

    validator = st.session_state.validator

    # Code input
    code_context = st.text_area("Enter your code context:", height=150,
                               value="def add(a, b):",  # Default example
                               key="code_input")
    
    # Debug information
    st.write("Debug Info:")
    st.write(f"Input code length: {len(code_context)}")
    st.write(f"Test mode: {validator.test_mode}")

    if code_context:
        st.subheader("Predictions")
        col1, col2 = st.columns(2)

        with col1:
            st.write("Next Token Prediction:")
            if st.button("Generate Next Token", key="gen_token"):
                try:
                    with st.spinner("Generating..."):
                        next_token = validator.generate_next_token(code_context)
                        st.write("Raw token:", repr(next_token))  # Show raw token for debugging
                        if next_token:
                            st.success("Token generated successfully!")
                            st.code(next_token, language="python")
                            st.write("Full code would be:")
                            st.code(code_context + next_token, language="python")
                        else:
                            st.warning("No token was generated")
                except Exception as e:
                    st.error(f"Error generating token: {str(e)}")

        with col2:
            st.write("Token Probabilities:")
            if st.button("Show Probabilities", key="show_probs"):
                try:
                    with st.spinner("Calculating probabilities..."):
                        probs = validator.get_next_token_probabilities(code_context, top_k=5)
                        if probs:
                            st.success("Top 5 probable next tokens:")
                            for token, prob in probs:
                                st.write(f"- '{token}' ({prob:.3f})")
                        else:
                            st.warning("No probabilities generated")
                except Exception as e:
                    st.error(f"Error calculating probabilities: {str(e)}")

        # Complete code generation section
        st.subheader("Complete Code Generation")
        temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.7, key="temp")
        if st.button("Generate Complete Code", key="gen_complete"):
            try:
                with st.spinner("Generating complete code..."):
                    completion = validator.generate_completion(code_context, temperature=temperature)
                    if completion:
                        st.success("Code generated successfully!")
                        st.code(completion, language="python")
                        st.write("Full code would be:")
                        st.code(code_context + completion, language="python")
                    else:
                        st.warning("No completion was generated")
            except Exception as e:
                st.error(f"Error generating complete code: {str(e)}")

if __name__ == "__main__":
    main() 