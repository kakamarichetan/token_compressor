import streamlit as st
import tiktoken
from sff_compressor.compressors import PromptCompressor
import logging
import torch

# Setup logging for debugging
logging.basicConfig(
    filename="app.log", 
    level=logging.DEBUG, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Function to compress prompt and calculate token savings
def compress_prompt_with_ratio(prompt: str, ratio: float):
    try:
        # Detect device (CPU or GPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize the compressor
        compressor = PromptCompressor(type='SCCompressor', model='gpt2', device=device)

        # Compress the original prompt
        compressed_prompt = compressor.compressgo(original_prompt=prompt, ratio=ratio)

        # Tokenize the original and compressed prompts
        tokenizer = tiktoken.get_encoding("gpt2")
        original_tokens = tokenizer.encode(prompt)
        compressed_tokens = tokenizer.encode(compressed_prompt)

        # Calculate the number of tokens saved
        original_token_count = len(original_tokens)
        compressed_token_count = len(compressed_tokens)

        # Calculate the percentage of tokens saved
        tokens_saved_percentage = (
            ((original_token_count - compressed_token_count) / original_token_count) * 100
            if original_token_count > 0 else 0.0
        )

        logging.info("Compression successful")
        return {
            "original_prompt": prompt,
            "compressed_prompt": compressed_prompt,
            "original_token_count": original_token_count,
            "compressed_token_count": compressed_token_count,
            "tokens_saved_percentage": tokens_saved_percentage
        }
    except Exception as e:
        logging.error(f"Error during compression: {str(e)}")
        raise

# Streamlit App UI
def main():
    # Streamlit Page Configuration
    st.set_page_config(page_title="SFF - Prompt Compressor", layout="centered")

    st.title("SFF - Prompt Compressor")
    st.subheader("Enter a Prompt and Compression Ratio")

    # Input fields for original prompt and ratio
    prompt_input = st.text_area("Original Prompt", height=200)
    ratio_input = st.slider("Compression Ratio", 0.0, 1.0, 0.4, step=0.05)

    # Button to trigger compression
    if st.button("Compress Prompt"):
        if not prompt_input:
            st.error("Please provide a prompt to compress!")
        else:
            try:
                # Compress the prompt and calculate token stats
                result = compress_prompt_with_ratio(prompt_input, ratio_input)

                # Display the results
                st.subheader("Compression Results")
                st.write(f"**Original Prompt:** {result['original_prompt']}")
                st.write(f"**Compressed Prompt:** {result['compressed_prompt']}")
                st.write(f"**Original Token Count:** {result['original_token_count']}")
                st.write(f"**Compressed Token Count:** {result['compressed_token_count']}")
                st.write(f"**Percentage of Tokens Saved:** {result['tokens_saved_percentage']:.2f}%")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logging.error(f"Error in Streamlit app: {str(e)}")

# Run the app
if __name__ == "__main__":
    main()
