# src/integrations/message_formatter.py
from src.models.data_models import CompleteHealthAnalysis, ImageValidationStatus

def format_whatsapp_message(report: CompleteHealthAnalysis) -> str:
    """
    Formats the complete health analysis into a single, beautiful string
    for WhatsApp, using its Markdown-like formatting.
    """
    if report.extracted_data.validation_status != ImageValidationStatus.VALID_FOOD_IMAGE:
        return f"⚠️ *Analysis Failed*\n\n{report.extracted_data.error_message}"

    product_name = report.extracted_data.product_name or "The Product"
    brand = f" (Brand: {report.extracted_data.brand})" if report.extracted_data.brand else ""

    # --- Header ---
    message_parts = [f"🍎 *Health Analysis for: {product_name}{brand}*"]

    # --- Overall Summary ---
    if report.overall_health_assessment:
        message_parts.append(f"\n*Overall Summary:*\n_{report.overall_health_assessment}_")

    message_parts.append("\n" + "-"*20)

    # --- Benefits Section ---
    if report.benefits_analysis and report.benefits_analysis.findings:
        message_parts.append("\n*✅ Key Benefits:*")
        for finding in report.benefits_analysis.findings[:3]: # Limit to top 3 for brevity
            message_parts.append(f"• {finding}")
    
    # --- Disadvantages Section ---
    if report.disadvantages_analysis and report.disadvantages_analysis.findings:
        message_parts.append("\n*⚠️ Key Concerns:*")
        for finding in report.disadvantages_analysis.findings[:3]:
            message_parts.append(f"• {finding}")

    # --- Disease Associations Section ---
    if report.disease_analysis and report.disease_analysis.findings:
        message_parts.append("\n*🩺 Potential Disease Associations:*")
        for finding in report.disease_analysis.findings[:3]:
            message_parts.append(f"• {finding}")

    # --- Alternatives Section ---
    if report.alternatives_report and report.alternatives_report.alternatives:
        message_parts.append(f"\n*❤️ Healthier Alternatives:*")
        for alt in report.alternatives_report.alternatives[:2]: # Limit to top 2 alternatives
            message_parts.append(f"• *{alt.product_name}:* _{alt.reason}_")
    
    # Join all parts with double newlines for clear separation
    return "\n\n".join(message_parts)

