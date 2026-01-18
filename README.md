# Docling PDF Extraction Service

A FastAPI microservice that uses IBM Docling to extract structured content from PDFs including:
- Text (plain and markdown formatted)
- Mathematical formulas (as LaTeX)
- Tables (as markdown and HTML)
- Images (with optional base64 encoding)

## Quick Start

### Option 1: Run Locally with Python

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the service
python main.py
```

### Option 2: Run with Docker

```bash
# Build the image
docker build -t docling-service .

# Run the container
docker run -p 8081:8081 -v /path/to/your/uploads:/app/uploads docling-service
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Extract from Uploaded File
```bash
POST /extract
Content-Type: multipart/form-data

file: <PDF file>
include_images_base64: false (optional)
document_id: <string> (optional)
```

### Extract from File Path (for files already on server)
```bash
POST /extract-from-path
Content-Type: application/x-www-form-urlencoded

file_path: /path/to/document.pdf
include_images_base64: false (optional)
document_id: <string> (optional)
```

### Get Extracted Image
```bash
GET /images/{image_id}
```

## Response Format

```json
{
  "document_id": "abc123",
  "total_pages": 10,
  "pages": [
    {
      "page_number": 1,
      "text": "Plain text content...",
      "markdown": "# Markdown content with $formulas$...",
      "formulas": [
        {
          "latex": "E = mc^2",
          "confidence": 0.95,
          "bbox": {"x": 100, "y": 200, "width": 50, "height": 20}
        }
      ],
      "tables": [
        {
          "markdown": "| Col1 | Col2 |\n|------|------|\n| A | B |",
          "html": "<table>...</table>",
          "rows": 2,
          "cols": 2
        }
      ],
      "images": [
        {
          "id": "abc123_p1_img0",
          "path": "/tmp/docling_images/abc123_p1_img0.png",
          "mime_type": "image/png",
          "width": 400,
          "height": 300,
          "caption": "Figure 1: Example"
        }
      ]
    }
  ],
  "full_markdown": "Complete document as markdown...",
  "full_text": "Complete document as plain text...",
  "metadata": {
    "title": "Document Title",
    "author": "Author Name"
  }
}
```

## Integration with Spring Boot

Configure the service URL in `application.properties`:
```properties
docling.service.url=http://localhost:8081
```
