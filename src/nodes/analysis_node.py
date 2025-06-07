# src/nodes/analysis_node.py
import time
from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.state.graph_state import HealthAdvisorState
from src.models.data_models import ExtractedIngredientsData, HealthAnalysisReport
from src.tools.search_tool import WebSearchTool

TEXT_ANALYSIS_MODEL = "compound-beta-mini" 

def _create_analysis_node_factory(
    analysis_type: str,
    system_prompt_template: str,
    human_prompt_template: str
):
    """
    A factory to create specific analysis node functions (benefits, disadvantages, disease).

    Args:
        analysis_type: The type of analysis (e.g., "benefits", "disadvantages", "disease_associations").
        system_prompt_template: The system prompt template for the LLM.
        human_prompt_template: The human prompt template for the LLM.
    Returns:
        A function that creates a node for the specified analysis type.
    """

    def create_node_function(groq_api_key: str):
        """
        Creates a node function for the specified analysis type.
        Args:
            groq_api_key: The API key for Groq.
        Returns:
            A function that processes the state and performs the analysis.
        """

        # Initialize the LLM and tools
        llm = ChatGroq(groq_api_key=groq_api_key, model_name=TEXT_ANALYSIS_MODEL, temperature=0.2)
        search_tool = WebSearchTool()
        parser = PydanticOutputParser(pydantic_object=HealthAnalysisReport)
        
        format_instructions = parser.get_format_instructions()
        # Escape curly braces to prevent them from being treated as template variables
        escaped_format_instructions = format_instructions.replace("{", "{{").replace("}", "}}")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_template.format(format_instructions=escaped_format_instructions, analysis_type=analysis_type)),
            ("human", human_prompt_template)
        ])
        
        chain = prompt | llm | parser

        def analysis_node(state: HealthAdvisorState) -> HealthAdvisorState:
            node_name = f"{analysis_type.capitalize()} Analysis Node"
            print(f"--- Running {node_name} ---")
            start_time = time.time()

            if state.get("should_stop_processing", False):
                print(f"--- Skipping {node_name} due to previous error/validation failure. ---")
                # Return only the field this node is responsible for
                return {f"{analysis_type}_analysis": HealthAnalysisReport(
                    analysis_type=analysis_type,
                    findings=["Analysis skipped"],
                    detailed_analysis=f"Processing stopped before {analysis_type} analysis.",
                    confidence_level="N/A",
                    health_score_impact=0
                )}

            extracted_data: Optional[ExtractedIngredientsData] = state.get("extracted_data")
            if not extracted_data or not extracted_data.ingredients:
                error_msg = f"No ingredients data available for {analysis_type} analysis."
                print(f"--- {node_name}: {error_msg} ---")
                return {f"{analysis_type}_analysis": HealthAnalysisReport(
                    analysis_type=analysis_type,
                    findings=["Missing ingredients"],
                    detailed_analysis=error_msg,
                    confidence_level="Low",
                    health_score_impact=0
                )}

            ingredients_list_str = ", ".join(extracted_data.ingredients)
            product_name = extracted_data.product_name or "the food product"
            
            # Perform a targeted web search for context
            search_query = f"{product_name} {ingredients_list_str[:100]} {analysis_type} effects"
            print(f"Performing web search for {analysis_type}: {search_query}")
            search_context = search_tool.search(query=search_query)

            try:
                report: HealthAnalysisReport = chain.invoke({
                    "product_name": product_name,
                    "ingredients_list": ingredients_list_str,
                    "allergens_list": ", ".join(extracted_data.allergens),
                    "nutritional_info_str": extracted_data.nutritional_info.model_dump_json() if extracted_data.nutritional_info else "Not available",
                    "search_context": search_context
                })
                
                # Return only the field this node is responsible for
                result = {f"{analysis_type}_analysis": report}

            except Exception as e:
                result = {f"{analysis_type}_analysis": HealthAnalysisReport(
                    analysis_type=analysis_type,
                    findings=[f"Error during {analysis_type} analysis"],
                    detailed_analysis=str(e),
                    confidence_level="Error",
                    health_score_impact=0
                )}
            
            processing_time = time.time() - start_time
            print(f"--- {node_name} completed in {processing_time:.2f}s ---")
            return result

        return analysis_node
    return create_node_function

# --- Benefits Analysis Node ---
BENEFITS_SYSTEM_PROMPT = """
You are a nutritional scientist AI. Your task is to analyze the provided food product details
and identify its potential health benefits.
Focus ONLY on positive aspects and scientifically plausible benefits.
Use the provided web search context for up-to-date information.
Respond STRICTLY in the following JSON format:
{format_instructions}
Ensure 'analysis_type' is '{analysis_type}'.
'health_score_impact' should be a positive number (0 to 10) if beneficial, 0 otherwise.
"""
BENEFITS_HUMAN_PROMPT = """
Product Name: {product_name}
Ingredients: {ingredients_list}
Allergens: {allergens_list}
Nutritional Information (JSON string): {nutritional_info_str}
Web Search Context: {search_context}

Please provide the health benefits analysis for this product.
"""
create_benefits_analysis_node = _create_analysis_node_factory(
    "benefits", BENEFITS_SYSTEM_PROMPT, BENEFITS_HUMAN_PROMPT
)

# --- Disadvantages Analysis Node ---
DISADVANTAGES_SYSTEM_PROMPT = """
You are a food safety and toxicology expert AI. Your task is to analyze the provided food product
details and identify potential health disadvantages, risks, or common concerns (e.g., allergens,
high sodium, artificial additives).
Focus ONLY on negative aspects and scientifically plausible risks.
Use the provided web search context for up-to-date information.
Respond STRICTLY in the following JSON format:
{format_instructions}
Ensure 'analysis_type' is '{analysis_type}'.
'health_score_impact' should be a negative number (-10 to 0) if detrimental, 0 otherwise.
"""
DISADVANTAGES_HUMAN_PROMPT = """
Product Name: {product_name}
Ingredients: {ingredients_list}
Allergens: {allergens_list}
Nutritional Information (JSON string): {nutritional_info_str}
Web Search Context: {search_context}

Please provide the health disadvantages/risks analysis for this product.
"""
create_disadvantages_analysis_node = _create_analysis_node_factory(
    "disadvantages", DISADVANTAGES_SYSTEM_PROMPT, DISADVANTAGES_HUMAN_PROMPT
)

# --- Disease Association Node ---
DISEASE_SYSTEM_PROMPT = """
You are a medical research AI specializing in epidemiology and nutritional science.
Analyze the list of ingredients and identify any known associations (positive or negative mitigation)
with common diseases or health conditions.
Base your analysis on reputable sources and the provided web search context. Present findings neutrally.
Respond STRICTLY in the following JSON format:
{format_instructions}
Ensure 'analysis_type' is '{analysis_type}'.
'health_score_impact' can be positive or negative (-5 to +5) based on disease risk modification, 0 if neutral.
"""
DISEASE_HUMAN_PROMPT = """
Product Name: {product_name}
Ingredients: {ingredients_list}
Allergens: {allergens_list}
Nutritional Information (JSON string): {nutritional_info_str}
Web Search Context: {search_context}

Please provide the ingredient-disease association analysis for this product.
"""
create_disease_analysis_node = _create_analysis_node_factory(
    "disease_associations", DISEASE_SYSTEM_PROMPT, DISEASE_HUMAN_PROMPT
)
