# src/nodes/analysis_nodes.py
from typing import List, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.state.graph_state import HealthAdvisorState
from src.models.data_models import HealthAnalysisReport
from src.prompts.analysis_prompts import (
    BENEFITS_SYSTEM_PROMPT, BENEFITS_HUMAN_PROMPT,
    DISADVANTAGES_SYSTEM_PROMPT, DISADVANTAGES_HUMAN_PROMPT,
    DISEASE_ASSOCIATIONS_SYSTEM_PROMPT, DISEASE_ASSOCIATIONS_HUMAN_PROMPT
)

TEXT_ANALYSIS_MODEL = "gemini-2.5-flash-preview-05-20"  

def _create_analysis_node_factory(
    analysis_type: str,
    system_prompt: str,
    human_prompt: str
):
    """A factory to create specialized analysis nodes using the ReAct framework with Gemini."""
    def create_node_function(google_api_key: str, mcp_tools: List[Any]):
        llm = ChatGoogleGenerativeAI(model=TEXT_ANALYSIS_MODEL, google_api_key=google_api_key, temperature=0.2)
        parser = PydanticOutputParser(pydantic_object=HealthAnalysisReport)
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt.format(format_instructions=parser.get_format_instructions().replace("{", "{{").replace("}", "}}"), analysis_type=analysis_type)),
            ("human", human_prompt),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        agent = create_tool_calling_agent(llm, mcp_tools, prompt_template)
        agent_executor = AgentExecutor(agent=agent, tools=mcp_tools, verbose=True)

        async def analysis_node(state: HealthAdvisorState) -> dict:
            node_name = f"{analysis_type.capitalize()} Analysis Node"
            print(f"--- Running {node_name} (ReAct + Gemini) ---")

            if state.get("should_stop_processing", False):
                return {f"{analysis_type}_analysis": HealthAnalysisReport(
                    analysis_type=analysis_type,
                    findings=["Analysis skipped due to prior error."],
                    detailed_analysis="Processing stopped before this step.",
                    confidence_level="N/A"
                )}

            extracted_data = state.get("extracted_data")
            if not extracted_data or not extracted_data.ingredients:
                return {f"{analysis_type}_analysis": HealthAnalysisReport(
                    analysis_type=analysis_type,
                    findings=["Missing ingredients"],
                    detailed_analysis="No ingredients data available for analysis.",
                    confidence_level="Low"
                )}

            input_data = {
                "product_name": extracted_data.product_name or "the food product",
                "ingredients_list": ", ".join(extracted_data.ingredients),
                "allergens_list": ", ".join(extracted_data.allergens),
                "nutritional_info_str": extracted_data.nutritional_info.model_dump_json() if extracted_data.nutritional_info else "Not available",
            }

            try:
                result = await agent_executor.ainvoke(input_data)
                final_llm_output = result.get("output", "{}")
                report = parser.parse(final_llm_output)
                return {f"{analysis_type}_analysis": report}

            except Exception as e:
                print(f"Error during {analysis_type} analysis: {e}")
                return {f"{analysis_type}_analysis": HealthAnalysisReport(
                    analysis_type=analysis_type,
                    findings=[f"An error occurred during the {analysis_type} analysis."],
                    detailed_analysis=str(e),
                    confidence_level="Error"
                )}

        return analysis_node
    return create_node_function

# Create the node functions using the factory and imported prompts
create_benefits_analysis_node = _create_analysis_node_factory(
    "benefits", BENEFITS_SYSTEM_PROMPT, BENEFITS_HUMAN_PROMPT
)

create_disadvantages_analysis_node = _create_analysis_node_factory(
    "disadvantages", DISADVANTAGES_SYSTEM_PROMPT, DISADVANTAGES_HUMAN_PROMPT
)

create_disease_analysis_node = _create_analysis_node_factory(
    "disease_analysis", DISEASE_ASSOCIATIONS_SYSTEM_PROMPT, DISEASE_ASSOCIATIONS_HUMAN_PROMPT
)
