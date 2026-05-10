from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_pdf():
    doc = SimpleDocTemplate("test_document.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Title style
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=14,
        alignment=1 # Center
    )
    
    body_style = styles['Normal']
    body_style.fontSize = 12
    body_style.spaceAfter = 12

    story = []
    
    story.append(Paragraph("Cognify Project Test Document", title_style))
    story.append(Spacer(1, 12))
    
    content = [
        "Introduction to Cognify Testing",
        "This is a sample document created specifically for testing the Cognify RAG (Retrieval-Augmented Generation) application. Cognify is designed to ingest documents, chunk them, embed them using OpenAI's embedding models, and store them in Qdrant for semantic search.",
        "Features of Cognify:",
        "1. Document Upload: Users can upload PDF documents up to 50MB.",
        "2. Vector Storage: Uses Qdrant for high-performance vector search.",
        "3. LLM Integration: Uses OpenAI's GPT models to generate answers based on retrieved context.",
        "4. Streaming Responses: Provides real-time typing effect via Server-Sent Events (SSE).",
        "System Architecture:",
        "The system uses FastAPI for the backend API, SQLAlchemy with PostgreSQL for relational metadata, Redis for caching and rate limiting, and Qdrant for the vector database. When a user uploads a document, the document is parsed, chunked according to the configured chunk size (1000) and overlap (200), and then embedded.",
        "Testing Scenarios:",
        "- Upload this document via the API.",
        "- Query: What is the maximum file size for upload?",
        "- Query: Which vector database does Cognify use?",
        "- Query: What technologies are used in the system architecture?",
        "End of Test Document."
    ]

    for text in content:
        story.append(Paragraph(text, body_style))
        story.append(Spacer(1, 6))

    doc.build(story)
    print("test_document.pdf generated successfully with ReportLab.")

if __name__ == "__main__":
    generate_pdf()
