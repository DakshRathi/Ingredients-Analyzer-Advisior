import time
from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.state.graph_state import HealthAdvisorState
from src.models.data_models import HealthyAlternativesReport, HealthAnalysisReport
from src.tools.search_tool import WebSearchTool

TEXT_ANALYSIS_MODEL = "compound-beta-mini"  

def create_alternatives_recommender_node(groq_api_key: str):
    """
    Factory function to create the healthy alternatives recommender node.
    """
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=TEXT_ANALYSIS_MODEL, temperature=0.3)
    search_tool = WebSearchTool()
    parser = PydanticOutputParser(pydantic_object=HealthyAlternativesReport)

    format_instructions = parser.get_format_instructions()
    escaped_format_instructions = format_instructions.replace("{", "{{").replace("}", "}}")

    system_prompt = f"""
    You are a registered dietitian and health-conscious food advisor AI.
    Based on the overall analysis of the original product's ingredients (benefits, disadvantages, disease associations),
    suggest 0 to 3 healthier alternative food items or specific ingredient substitutions.
    If the original product is generally healthy, you can state that and suggest no alternatives or minor improvements.
    Alternatives should be generally accessible and practical. Use web search for ideas if needed.

    Respond STRICTLY in the following JSON format:
    {escaped_format_instructions}
    Provide a concise 'summary' explaining your overall recommendation.
    The 'alternatives' list can be empty if the product is deemed healthy enough.
    CRITICAL: Respond with ONLY valid JSON. Do not include any explanatory text, introductions, or conclusions outside the JSON.
    """

    human_prompt_template = """
    Original Product Analysis:
    Product Name: {product_name}
    Ingredients: {ingredients_list}

    Benefits Analysis Summary:
    {benefits_summary}

    Disadvantages Analysis Summary:
    {disadvantages_summary}

    Disease Associations Summary:
    {disease_summary}

    Web Search Context for Alternatives: {search_context}

    Please provide healthier alternative recommendations based on this comprehensive analysis.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt_template)
    ])
    
    chain = prompt | llm | parser

    def alternatives_recommender_node(state: HealthAdvisorState) -> HealthAdvisorState:
        print("--- Running Alternatives Recommender Node ---")
        state['current_task_start_time'] = time.time()

        if state.get("should_stop_processing", False):
            print("--- Skipping Alternatives Recommender due to previous error/validation failure. ---")
            state["alternatives_report"] = HealthyAlternativesReport(
                summary="Recommendation skipped due to issues in prior analysis.",
                alternatives=[]
            )
            return state

        extracted_data = state.get("extracted_data")
        benefits_report: Optional[HealthAnalysisReport] = state.get("benefits_analysis")
        disadvantages_report: Optional[HealthAnalysisReport] = state.get("disadvantages_analysis")
        disease_report: Optional[HealthAnalysisReport] = state.get("disease_analysis")

        if not extracted_data:
            state["alternatives_report"] = HealthyAlternativesReport(
                summary="Cannot recommend alternatives without ingredient data.",
                alternatives=[]
            )
            return state
        
        # Prepare summaries for the prompt
        benefits_summary = benefits_report.detailed_analysis if benefits_report else "Not analyzed."
        disadvantages_summary = disadvantages_report.detailed_analysis if disadvantages_report else "Not analyzed."
        disease_summary = disease_report.detailed_analysis if disease_report else "Not analyzed."

        product_name = extracted_data.product_name or "the food product"
        ingredients_list_str = ", ".join(extracted_data.ingredients)

        # Search for alternatives based on product type or key ingredients
        search_query = f"healthy alternatives for {product_name} with {ingredients_list_str[:50]}"
        print(f"Performing web search for alternatives: {search_query}")
        search_context = search_tool.search(query=search_query) # Concise search context
        # print(f"Search context for alternatives:\n{search_context}")

        try:
            report: HealthyAlternativesReport = chain.invoke({
                "product_name": product_name,
                "ingredients_list": ingredients_list_str,
                "benefits_summary": benefits_summary,
                "disadvantages_summary": disadvantages_summary,
                "disease_summary": disease_summary,
                "search_context": search_context
            })
            state["alternatives_report"] = report
            # print(f"Alternatives Report:\n{report.model_dump_json(indent=2)}")

        except Exception as e:
            # print(f"Error during alternatives recommendation: {e}")
            state["alternatives_report"] = HealthyAlternativesReport(
                summary=f"Error generating alternatives: {str(e)}",
                alternatives=[]
            )
        
        processing_time = time.time() - state['current_task_start_time']
        print(f"--- Alternatives Recommender Node completed in {processing_time:.2f}s ---")
        return state
    return alternatives_recommender_node
