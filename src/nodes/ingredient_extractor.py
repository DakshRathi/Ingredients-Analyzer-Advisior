# src/nodes/ingredient_extractor.py
import base64
import time
from groq import Groq
from langchain_core.output_parsers import PydanticOutputParser

from src.state.graph_state import HealthAdvisorState
from src.models.data_models import ExtractedIngredientsData, ImageValidationStatus

def create_ingredient_extractor_node(groq_api_key: str):
    """
    Factory function to create the ingredient extraction node.
    This node uses a Groq Vision LLM to validate the image and extract ingredient data.
    """
    groq_client = Groq(api_key=groq_api_key)
    # The parser expects the LLM to output JSON that matches the ExtractedIngredientsData schema.
    parser = PydanticOutputParser(pydantic_object=ExtractedIngredientsData)
    vision_model_name = "meta-llama/llama-4-scout-17b-16e-instruct" # Groq's vision model

    def _encode_image_to_base64(image_path: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found at {image_path}")
        except Exception as e:
            raise RuntimeError(f"Error encoding image {image_path}: {e}")

    def ingredient_extractor_node(state: HealthAdvisorState) -> HealthAdvisorState:
        """
        Processes an image to extract food ingredient information.
        Validates if the image is a food product with ingredients before full extraction.
        """
        print("--- Running Ingredient Extractor Node ---")
        state['current_task_start_time'] = time.time()
        image_path = state["image_path"]

        try:
            base64_image = _encode_image_to_base64(image_path)
        except Exception as e:
            error_data = ExtractedIngredientsData(
                validation_status=ImageValidationStatus.ERROR,
                is_food_product=False,
                confidence_score=0.0, # Explicitly set confidence
                error_message=str(e)
            )
            state["extracted_data"] = error_data
            state["should_stop_processing"] = True
            state["error_message"] = str(e)
            return state

        # PydanticOutputParser provides format instructions for the LLM.
        format_instructions = parser.get_format_instructions()

        # Prompt designed to guide the LLM for both validation and extraction.
        # The LLM should fill the fields of ExtractedIngredientsData.
        prompt_text = f"""
        You are an expert food label analyzer. Analyze the provided image.
        First, determine if this image contains a food product with a visible ingredients list.
        - If it's not a food product, set 'is_food_product' to false, 'validation_status' to '{ImageValidationStatus.INVALID_NOT_FOOD.value}', and provide a brief 'error_message'.
        - If it's a food product but no ingredients list is visible or readable, set 'is_food_product' to true, 'validation_status' to '{ImageValidationStatus.INVALID_NO_INGREDIENTS.value}', and provide an 'error_message'.
        - If the image quality is too poor for reliable analysis, set 'validation_status' to '{ImageValidationStatus.INVALID_POOR_QUALITY.value}' and provide an 'error_message'.
        - If it's a valid food product with a readable ingredients list, set 'is_food_product' to true, 'validation_status' to '{ImageValidationStatus.VALID_FOOD_IMAGE.value}', and proceed to extract all information.

        If valid, extract the following: product_name, brand, ingredients (as a list of strings),
        allergens (as a list of strings from 'Contains' or similar statements),
        warnings (e.g., 'high in sodium'), and nutritional_info (per 100g if available, otherwise leave fields null).
        Set 'confidence_score' from 0.0 to 1.0 based on your certainty.

        Respond STRICTLY in the following JSON format:
        {format_instructions}
        """

        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
                model=vision_model_name,
                temperature=0.1, # Low temperature for factual extraction
                max_tokens=2048, # Increased for potentially long ingredient lists & schema
                response_format={"type": "json_object"} # Ensure JSON output
            )
            llm_output_json_str = chat_completion.choices[0].message.content
            # print(f"LLM Raw Output for Image Extraction:\n{llm_output_json_str}") # For debugging

            # Parse the LLM's JSON output using the PydanticOutputParser
            extracted_data: ExtractedIngredientsData = parser.parse(llm_output_json_str)
            
            state["extracted_data"] = extracted_data
            state["should_stop_processing"] = extracted_data.validation_status != ImageValidationStatus.VALID_FOOD_IMAGE
            if state["should_stop_processing"]:
                state["error_message"] = extracted_data.error_message or "Image validation failed."
            
            # print(f"Parsed Extracted Data: {extracted_data.model_dump_json(indent=2)}") # For debugging

        except Exception as e:
            # print(f"Error during LLM call or parsing in ingredient_extractor_node: {e}") # For debugging
            error_data = ExtractedIngredientsData(
                validation_status=ImageValidationStatus.ERROR,
                is_food_product=False, # Assume false on error
                confidence_score=0.0,
                error_message=f"LLM processing or parsing error: {str(e)}"
            )
            state["extracted_data"] = error_data
            state["should_stop_processing"] = True
            state["error_message"] = str(e)
        
        processing_time = time.time() - state['current_task_start_time']
        print(f"--- Ingredient Extractor Node completed in {processing_time:.2f}s ---")
        return state

    return ingredient_extractor_node

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    print(__name__, "is running as a standalone script.")
    # Example usage (optional, for testing the node directly)
    load_dotenv()
    test_state = HealthAdvisorState(
        image_path="/Users/daksh/Desktop/Health/test_img.jpg",
        current_task_start_time=0,
        extracted_data=None,
        should_stop_processing=False,
        error_message=""
    )
    extractor_node = create_ingredient_extractor_node(groq_api_key=os.environ.get("GROQ_API_KEY"))
    result_state = extractor_node(test_state)
    print(f"Result State: {result_state}")