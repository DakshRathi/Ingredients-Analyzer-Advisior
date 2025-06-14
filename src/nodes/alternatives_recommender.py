# src/nodes/alternatives_recommender.py
import time
from typing import Optional, List, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.state.graph_state import HealthAdvisorState
from src.models.data_models import HealthyAlternativesReport, HealthAnalysisReport
from src.prompts.alternatives_prompt import SYSTEM_PROMPT, HUMAN_PROMPT

TEXT_ANALYSIS_MODEL = "gemini-2.5-flash-preview-05-20"  

def create_alternatives_recommender_node(api_key: str, mcp_tools: List[Any]) -> callable:
    """
    Factory function to create the healthy alternatives recommender node.
    """

    llm = ChatGoogleGenerativeAI(model=TEXT_ANALYSIS_MODEL, google_api_key=api_key, temperature=0.2)
    parser = PydanticOutputParser(pydantic_object=HealthyAlternativesReport)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
        ("placeholder", "{agent_scratchpad}"),
    ]).partial(format_instructions=parser.get_format_instructions().replace("{", "{{").replace("}", "}}"))

    agent = create_tool_calling_agent(llm, mcp_tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=mcp_tools, verbose=True)

    async def alternatives_recommender_node(state: HealthAdvisorState) -> dict:
        print("--- Running ReAct Alternatives Recommender Node ---")
        start_time = time.time()

        if state.get("should_stop_processing", False):
            return {"alternatives_report": HealthyAlternativesReport(summary="Recommendation skipped.", alternatives=[])}

        extracted_data = state.get("extracted_data")
        if not extracted_data:
            return {"alternatives_report": HealthyAlternativesReport(summary="Cannot recommend without data.", alternatives=[])}

        benefits_report: Optional[HealthAnalysisReport] = state.get("benefits_analysis")
        disadvantages_report: Optional[HealthAnalysisReport] = state.get("disadvantages_analysis")
        disease_report: Optional[HealthAnalysisReport] = state.get("disease_analysis")

        input_data = {
            "product_name": extracted_data.product_name or "the food product",
            "ingredients_list": ", ".join(extracted_data.ingredients),
            "benefits_summary": benefits_report.detailed_analysis if benefits_report else "Not analyzed.",
            "disadvantages_summary": disadvantages_report.detailed_analysis if disadvantages_report else "Not analyzed.",
            "disease_summary": disease_report.detailed_analysis if disease_report else "Not analyzed.",
        }

        try:
            result = await agent_executor.ainvoke(input_data)
            final_llm_output = result.get("output", "{}")
            report = parser.parse(final_llm_output)
            print(f"Alternatives report generated in {time.time() - start_time:.2f} seconds")
            return {"alternatives_report": report}
        except Exception as e:
            print(f"Error during ReAct alternatives recommendation: {e}")
            return {"alternatives_report": HealthyAlternativesReport(summary=f"Error generating alternatives: {e}", alternatives=[])}

    return alternatives_recommender_node