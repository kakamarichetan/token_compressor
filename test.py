from sff_compressor.compressors import PromptCompressor
import tiktoken
import unittest

test_prompt = """### Task
A person is going on a road trip with their family members, and before setting off, they each introduce themselves by sharing their name, age, relationship to the family, a couple of physical traits, a few personality attributes, and their favorite travel accessory for the trip.
Your task is to extract the following details from the input text based on the introduction, without guessing any missing information:

1. **Name**: Look for names provided directly.
2. **Age**: Extract if explicitly mentioned. If age is not provided, estimate it based on the context or age type, if possible.
3. **Age Type**: Identify whether the person is "younger" (typically under 30), "middle-aged" (30-60), or "older" (60+), based on the age provided. If an explicit age is mentioned, use it to determine the age type. Do not infer the age type from relative terms like "elder son" or "older dad," as these terms refer to family roles, not the person's actual age_type.
4. **Gender**: Identify the gender if mentioned directly (either male or female) or implicitly find from the context or with the help of name.
5. **Relation**: Extract the familial relationship (e.g., dad, mom, brother, sister, etc.).
6. **Personality Traits**: Identify any personality traits or preferences shared, prioritizing core personality traits (e.g., independent, ambitious) over context-specific traits or interests (e.g., animal lover, road trip leader), including both direct mentions (e.g., jovial, traveller, gardener) and implicit traits based on hobbies, or statements. When extracting personality traits, keep compound phrases (e.g., "well-seasoned road trip leader") intact rather than breaking them into individual words. Only include up to three strong traits, prioritizing the most significant ones mentioned. Do not split traits into smaller descriptive words like "great" or "well-seasoned" if they are part of a broader, meaningful phrase.
7. **Favorite Travel Accessory**: Extract the travel accessory mentioned, if any (e.g., skateboard, hat, camera, etc.).

### Output Format
Format the output as a JSON object with the following keys. 
{{
    "name": "", # leave empty if not provided
    "age": "", # leave empty if not provided
    "age_type": "", # leave empty if not inferable
    "gender": "", # leave empty if not mentioned
    "relation": "", # leave empty if not mentioned
    "personality_traits": [], # leave as empty list if not mentioned
    "travel_accessory": "" # leave empty if not mentioned
}}
- Ensure all keys are present in the output, even if their values are empty or not provided.
Introduction Input Text: {input}
"""

class TestPromptCompressor(unittest.TestCase):

    def setUp(self):
        # Initialize the PromptCompressor with desired settings
        self.compressor = PromptCompressor(type='SCCompressor', model='gpt2',device="cpu")

    def test_compress_prompt(self):
        # Sample prompt to test compression
        sample_prompt = """
        This is a test sentence. We want to compress this prompt to see how the compressor works. The sentence includes some key information that should be preserved, but reduced in size to save tokens.

        We should test this compression at a ratio to see how well it reduces the content while keeping important context.
        """

        # Compression ratio set
        ratio = 0.3

        # Call the compressgo method to compress the sample prompt
        compressed_prompt = self.compressor.compressgo(original_prompt=sample_prompt, ratio=ratio)

        # Output the compressed prompt for verification
        print("Original Prompt:")
        print(sample_prompt)
        print("\nCompressed Prompt:")
        print(compressed_prompt)

        # Verify that the compressed prompt is not empty and has reduced size
        self.assertTrue(compressed_prompt)
        self.assertLess(len(compressed_prompt), len(sample_prompt),
                        "The compressed prompt should be smaller than the original one.")

        # Optionally, you can check if token length matches expectations using tiktoken tokenizer
        # Tokenize the original and compressed prompt
        tokenizer = tiktoken.get_encoding("gpt2")
        original_tokens = tokenizer.encode(sample_prompt)
        compressed_tokens = tokenizer.encode(compressed_prompt)

        # Output the token counts for verification
        print(f"\nOriginal token count: {len(original_tokens)}")
        print(f"Compressed token count: {len(compressed_tokens)}")

        # Verify the token reduction is as expected
        self.assertLess(len(compressed_tokens), len(original_tokens), "The compressed prompt should have fewer tokens.")

        print("% of Tokens saved",float((len(original_tokens)-len(compressed_tokens))/len(original_tokens))*100)

if __name__ == "__main__":
    unittest.main()
