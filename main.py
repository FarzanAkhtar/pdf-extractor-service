"""
Docling PDF Extraction Microservice
Extracts text, formulas (LaTeX), images, and tables from PDFs.
"""

import base64
import hashlib
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, 
    TableFormerMode,
    EasyOcrOptions,
    OcrOptions,
)
from docling.document_converter import PdfFormatOption

# Try to import equation pipeline options (may vary by docling version)
try:
    from docling.datamodel.pipeline_options import EquationPipelineOptions
    HAS_EQUATION_PIPELINE = True
except ImportError:
    HAS_EQUATION_PIPELINE = False
    print("Warning: EquationPipelineOptions not available in this docling version")

app = FastAPI(
    title="Docling PDF Extraction Service",
    description="Extracts structured content from PDFs including text, formulas, tables, and images",
    version="1.0.0"
)

# Configure storage for extracted images
IMAGES_DIR = Path(os.getenv("DOCLING_IMAGES_DIR", "/tmp/docling_images"))
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


class Formula(BaseModel):
    """Extracted formula with LaTeX representation"""
    latex: str
    confidence: Optional[float] = None
    bbox: Optional[dict] = None  # {x, y, width, height}


class Table(BaseModel):
    """Extracted table with markdown and optional HTML representation"""
    markdown: str
    html: Optional[str] = None
    rows: int
    cols: int
    bbox: Optional[dict] = None


class Image(BaseModel):
    """Extracted image metadata"""
    id: str
    path: str  # Local path or URL where image is stored
    base64_data: Optional[str] = None  # Base64 encoded image data (optional)
    mime_type: str
    width: Optional[int] = None
    height: Optional[int] = None
    bbox: Optional[dict] = None
    caption: Optional[str] = None


class PageContent(BaseModel):
    """Content extracted from a single page"""
    page_number: int
    text: str  # Plain text content
    markdown: str  # Markdown formatted content with formulas
    formulas: list[Formula] = []
    tables: list[Table] = []
    images: list[Image] = []


class ExtractionResult(BaseModel):
    """Complete extraction result for a document"""
    document_id: str
    total_pages: int
    pages: list[PageContent]
    full_markdown: str  # Complete document as markdown
    full_text: str  # Complete document as plain text
    metadata: dict = {}


class HealthResponse(BaseModel):
    status: str
    version: str


def get_converter() -> DocumentConverter:
    """Create configured DocumentConverter instance with formula extraction enabled"""
    pipeline_options = PdfPipelineOptions()
    
    # Enable OCR for scanned documents
    pipeline_options.do_ocr = True
    
    # Configure OCR options
    try:
        ocr_options = EasyOcrOptions(
            lang=["en"],
            use_gpu=False,  # Set to True if GPU available
        )
        pipeline_options.ocr_options = ocr_options
    except Exception as e:
        print(f"Warning: Could not configure EasyOCR options: {e}")
    
    # Enable table structure recognition
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    
    # Enable image extraction
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    
    # Enable formula/equation extraction - THIS IS KEY
    # In newer docling versions, this enables LaTeX conversion
    if hasattr(pipeline_options, 'do_formula_enrichment'):
        pipeline_options.do_formula_enrichment = True
        print("Formula enrichment enabled via do_formula_enrichment")
    
    if hasattr(pipeline_options, 'do_code_enrichment'):
        pipeline_options.do_code_enrichment = True
    
    # Try to enable equation pipeline if available
    if HAS_EQUATION_PIPELINE:
        try:
            pipeline_options.equation_pipeline_options = EquationPipelineOptions(
                enabled=True
            )
            print("Equation pipeline enabled")
        except Exception as e:
            print(f"Warning: Could not enable equation pipeline: {e}")
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    return converter


def _extract_bbox(item) -> Optional[dict]:
    """Extract bounding box from an item if available"""
    try:
        if hasattr(item, 'prov') and item.prov:
            # New Docling API uses prov for provenance/location info
            for prov in item.prov:
                if hasattr(prov, 'bbox'):
                    bbox = prov.bbox
                    return {
                        "x": getattr(bbox, 'l', 0),
                        "y": getattr(bbox, 't', 0),
                        "width": getattr(bbox, 'r', 0) - getattr(bbox, 'l', 0),
                        "height": getattr(bbox, 'b', 0) - getattr(bbox, 't', 0)
                    }
        if hasattr(item, 'bbox'):
            bbox = item.bbox
            return {
                "x": getattr(bbox, 'x', getattr(bbox, 'l', 0)),
                "y": getattr(bbox, 'y', getattr(bbox, 't', 0)),
                "width": getattr(bbox, 'width', getattr(bbox, 'r', 0) - getattr(bbox, 'l', 0)),
                "height": getattr(bbox, 'height', getattr(bbox, 'b', 0) - getattr(bbox, 't', 0))
            }
    except Exception:
        pass
    return None


def _get_page_number(item) -> int:
    """Extract page number from an item's provenance"""
    try:
        if hasattr(item, 'prov') and item.prov:
            for prov in item.prov:
                if hasattr(prov, 'page_no'):
                    return prov.page_no
                if hasattr(prov, 'page'):
                    return prov.page
    except Exception:
        pass
    return 1


def extract_content_from_document(doc, doc_id: str, include_base64: bool = False) -> tuple[list[Formula], list[Table], list[Image]]:
    """Extract formulas, tables, and images from document by iterating items"""
    all_formulas = []
    all_tables = []
    all_images = []
    image_idx = 0
    
    try:
        # Try iterate_items first (newer Docling API)
        items_iterator = None
        if hasattr(doc, 'iterate_items'):
            items_iterator = doc.iterate_items()
        elif hasattr(doc, 'items'):
            items_iterator = doc.items
        
        if items_iterator:
            for item in items_iterator:
                # Handle tuple (element, level) format
                element = item[0] if isinstance(item, tuple) else item
                
                # Get label safely
                label = ""
                if hasattr(element, 'label'):
                    label = str(element.label).lower()
                elif hasattr(element, '__class__'):
                    label = element.__class__.__name__.lower()
                
                page_num = _get_page_number(element)
                
                # Check for equations/formulas
                if 'equation' in label or 'formula' in label or 'math' in label:
                    latex = ""
                    if hasattr(element, 'text'):
                        latex = element.text or ""
                    elif hasattr(element, 'export_to_markdown'):
                        try:
                            latex = element.export_to_markdown()
                        except:
                            pass
                    
                    if latex and latex.strip():
                        all_formulas.append(Formula(
                            latex=latex.strip(),
                            confidence=getattr(element, 'confidence', None),
                            bbox=_extract_bbox(element)
                        ))
                
                # Check for tables
                elif 'table' in label:
                    md = ""
                    html = None
                    rows = 0
                    cols = 0
                    
                    if hasattr(element, 'export_to_markdown'):
                        try:
                            md = element.export_to_markdown()
                        except:
                            pass
                    if hasattr(element, 'export_to_html'):
                        try:
                            html = element.export_to_html()
                        except:
                            pass
                    if hasattr(element, 'num_rows'):
                        rows = element.num_rows or 0
                    if hasattr(element, 'num_cols'):
                        cols = element.num_cols or 0
                    
                    # Try to get table data for row/col count
                    if hasattr(element, 'data') and element.data:
                        try:
                            if hasattr(element.data, 'num_rows'):
                                rows = element.data.num_rows
                            if hasattr(element.data, 'num_cols'):
                                cols = element.data.num_cols
                        except:
                            pass
                    
                    if md and md.strip():
                        all_tables.append(Table(
                            markdown=md,
                            html=html,
                            rows=rows,
                            cols=cols,
                            bbox=_extract_bbox(element)
                        ))
                
                # Check for pictures/images
                elif 'picture' in label or 'image' in label or 'figure' in label:
                    image_id = f"{doc_id}_p{page_num}_img{image_idx}"
                    image_path = IMAGES_DIR / f"{image_id}.png"
                    image_idx += 1
                    
                    base64_data = None
                    mime_type = "image/png"
                    width = None
                    height = None
                    caption = None
                    image_saved = False
                    
                    # Try to get image data - handle different Docling API versions
                    try:
                        img_obj = None
                        
                        # Method 1: Check for 'image' attribute (ImageRef in newer versions)
                        if hasattr(element, 'image') and element.image is not None:
                            img_ref = element.image
                            
                            # ImageRef has 'pil_image' property or 'get_image()' method
                            if hasattr(img_ref, 'pil_image'):
                                img_obj = img_ref.pil_image
                            elif hasattr(img_ref, 'get_image'):
                                img_obj = img_ref.get_image()
                            elif hasattr(img_ref, 'image'):
                                img_obj = img_ref.image
                            # If it's already a PIL image
                            elif hasattr(img_ref, 'save'):
                                img_obj = img_ref
                        
                        # Method 2: Check for 'prov' with image data
                        if img_obj is None and hasattr(element, 'prov'):
                            for prov in element.prov:
                                if hasattr(prov, 'image') and prov.image is not None:
                                    img_ref = prov.image
                                    if hasattr(img_ref, 'pil_image'):
                                        img_obj = img_ref.pil_image
                                    elif hasattr(img_ref, 'get_image'):
                                        img_obj = img_ref.get_image()
                                    break
                        
                        # Save the image if we got it
                        if img_obj is not None and hasattr(img_obj, 'save'):
                            img_obj.save(str(image_path))
                            width, height = img_obj.size
                            image_saved = True
                            
                            if include_base64:
                                with open(image_path, "rb") as f:
                                    base64_data = base64.b64encode(f.read()).decode('utf-8')
                                    
                    except Exception as img_err:
                        print(f"Warning: Could not save image: {img_err}")
                    
                    # Try to get caption
                    if hasattr(element, 'caption'):
                        caption = str(element.caption) if element.caption else None
                    elif hasattr(element, 'text'):
                        caption = str(element.text) if element.text else None
                    
                    # Only add image if we have some useful info
                    all_images.append(Image(
                        id=image_id,
                        path=str(image_path) if image_saved else "",
                        base64_data=base64_data,
                        mime_type=mime_type,
                        width=width,
                        height=height,
                        bbox=_extract_bbox(element),
                        caption=caption
                    ))
                    
    except Exception as e:
        print(f"Warning: Error extracting structured content: {e}")
    
    return all_formulas, all_tables, all_images


def extract_formulas_from_markdown(markdown: str) -> list[Formula]:
    """Extract LaTeX formulas from markdown text"""
    formulas = []
    
    # Match display math: $$...$$ or \[...\]
    display_patterns = [
        r'\$\$(.+?)\$\$',
        r'\\\[(.+?)\\\]'
    ]
    
    # Match inline math: $...$ or \(...\)
    inline_patterns = [
        r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)',
        r'\\\((.+?)\\\)'
    ]
    
    for pattern in display_patterns + inline_patterns:
        matches = re.findall(pattern, markdown, re.DOTALL)
        for match in matches:
            latex = match.strip()
            if latex and len(latex) > 1:  # Skip single characters
                formulas.append(Formula(latex=latex))
    
    return formulas


def extract_equations_from_doc(doc) -> list[Formula]:
    """Extract equations directly from document structure"""
    formulas = []
    
    try:
        # Method 1: Check for equations in document body
        if hasattr(doc, 'body'):
            body = doc.body
            if hasattr(body, 'children'):
                for child in body.children:
                    _extract_equation_recursive(child, formulas)
        
        # Method 2: Check document's equations list if available
        if hasattr(doc, 'equations'):
            for eq in doc.equations:
                latex = None
                if hasattr(eq, 'latex'):
                    latex = eq.latex
                elif hasattr(eq, 'text'):
                    latex = eq.text
                elif hasattr(eq, 'original'):
                    latex = eq.original
                
                if latex and latex.strip():
                    formulas.append(Formula(
                        latex=latex.strip(),
                        bbox=_extract_bbox(eq)
                    ))
        
        # Method 3: Check for formula items in main_text
        if hasattr(doc, 'main_text'):
            for item in doc.main_text:
                if hasattr(item, 'obj'):
                    obj = item.obj
                    obj_type = type(obj).__name__.lower()
                    if 'equation' in obj_type or 'formula' in obj_type:
                        latex = getattr(obj, 'latex', None) or getattr(obj, 'text', None)
                        if latex:
                            formulas.append(Formula(latex=latex.strip()))
                            
    except Exception as e:
        print(f"Warning: Error extracting equations from doc structure: {e}")
    
    return formulas


def _extract_equation_recursive(element, formulas: list):
    """Recursively extract equations from document elements"""
    try:
        elem_type = type(element).__name__.lower()
        
        # Check if this element is an equation
        if 'equation' in elem_type or 'formula' in elem_type or 'math' in elem_type:
            latex = None
            if hasattr(element, 'latex'):
                latex = element.latex
            elif hasattr(element, 'text'):
                latex = element.text
            elif hasattr(element, 'original'):
                latex = element.original
            
            if latex and latex.strip() and '<!-- formula-not-decoded -->' not in latex:
                formulas.append(Formula(
                    latex=latex.strip(),
                    bbox=_extract_bbox(element)
                ))
        
        # Recurse into children
        if hasattr(element, 'children'):
            for child in element.children:
                _extract_equation_recursive(child, formulas)
                
    except Exception as e:
        print(f"Warning: Error in recursive equation extraction: {e}")


def process_document(doc, doc_id: str, include_base64: bool = False) -> ExtractionResult:
    """Process a Docling document and extract all content"""
    import time
    start_time = time.time()
    
    print(f"[{doc_id}] Starting document processing...")
    
    # Get total pages
    total_pages = 1
    if hasattr(doc, 'pages'):
        total_pages = len(doc.pages) if doc.pages else 1
    
    print(f"[{doc_id}] Document has {total_pages} pages")
    
    # Export full document content
    full_text = ""
    full_markdown = ""
    
    try:
        print(f"[{doc_id}] Exporting text...")
        if hasattr(doc, 'export_to_text'):
            full_text = doc.export_to_text()
        print(f"[{doc_id}] Text export done: {len(full_text)} chars ({time.time() - start_time:.1f}s)")
    except Exception as e:
        print(f"Warning: Error exporting text: {e}")
    
    try:
        print(f"[{doc_id}] Exporting markdown...")
        if hasattr(doc, 'export_to_markdown'):
            full_markdown = doc.export_to_markdown()
        print(f"[{doc_id}] Markdown export done: {len(full_markdown)} chars ({time.time() - start_time:.1f}s)")
    except Exception as e:
        print(f"Warning: Error exporting markdown: {e}")
    
    # Extract structured elements from document items
    print(f"[{doc_id}] Extracting structured elements...")
    all_formulas, all_tables, all_images = extract_content_from_document(doc, doc_id, include_base64)
    print(f"[{doc_id}] Structured extraction done ({time.time() - start_time:.1f}s)")
    
    # Also try extracting equations directly from document structure
    doc_equations = extract_equations_from_doc(doc)
    existing_latex = {f.latex for f in all_formulas}
    for f in doc_equations:
        if f.latex not in existing_latex:
            all_formulas.append(f)
            existing_latex.add(f.latex)
    
    # Also extract formulas from markdown (catches any missed by item iteration)
    markdown_formulas = extract_formulas_from_markdown(full_markdown)
    for f in markdown_formulas:
        if f.latex not in existing_latex:
            all_formulas.append(f)
            existing_latex.add(f.latex)
    
    # Log formula extraction results
    print(f"Extracted {len(all_formulas)} formulas, {len(all_tables)} tables, {len(all_images)} images")
    
    # Check if formulas were not decoded
    if '<!-- formula-not-decoded -->' in full_markdown:
        print("WARNING: Some formulas could not be decoded to LaTeX.")
        print("To enable formula decoding, ensure you have installed:")
        print("  pip install 'docling[ocr]' easyocr torch torchvision")
        print("Or try: pip install docling-ibm-models")
    
    print(f"[{doc_id}] Document processing complete ({time.time() - start_time:.1f}s)")
    
    # Create page content - Docling doesn't easily provide per-page breakdown
    # so we put all content in page 1 and use the markdown for LLM processing
    pages = []
    
    if total_pages == 1 or not full_markdown:
        # Single page or simple document
        pages.append(PageContent(
            page_number=1,
            text=full_text,
            markdown=full_markdown,
            formulas=all_formulas,
            tables=all_tables,
            images=all_images
        ))
    else:
        # Try to split content by pages if possible
        # For now, put all structured content on first page
        # but distribute text/markdown evenly
        text_per_page = len(full_text) // total_pages if full_text else 0
        md_per_page = len(full_markdown) // total_pages if full_markdown else 0
        
        for page_num in range(1, total_pages + 1):
            start_text = (page_num - 1) * text_per_page
            end_text = page_num * text_per_page if page_num < total_pages else len(full_text)
            
            start_md = (page_num - 1) * md_per_page
            end_md = page_num * md_per_page if page_num < total_pages else len(full_markdown)
            
            page_content = PageContent(
                page_number=page_num,
                text=full_text[start_text:end_text] if full_text else "",
                markdown=full_markdown[start_md:end_md] if full_markdown else "",
                formulas=all_formulas if page_num == 1 else [],
                tables=all_tables if page_num == 1 else [],
                images=all_images if page_num == 1 else []
            )
            pages.append(page_content)
    
    # Extract metadata
    metadata = {}
    try:
        if hasattr(doc, 'metadata') and doc.metadata:
            for key in ['title', 'author', 'subject', 'keywords', 'creator', 'producer']:
                val = getattr(doc.metadata, key, None)
                if val:
                    metadata[key] = str(val)
    except Exception:
        pass
    
    return ExtractionResult(
        document_id=doc_id,
        total_pages=total_pages,
        pages=pages,
        full_markdown=full_markdown,
        full_text=full_text,
        metadata=metadata
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy", version="1.0.0")


@app.post("/extract", response_model=ExtractionResult)
async def extract_pdf(
    file: UploadFile = File(...),
    include_images_base64: bool = Form(default=False),
    document_id: Optional[str] = Form(default=None)
):
    """
    Extract content from a PDF file.
    
    - **file**: PDF file to process
    - **include_images_base64**: If true, include base64 encoded image data in response
    - **document_id**: Optional document ID (will be generated if not provided)
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Generate document ID if not provided
    doc_id = document_id or str(uuid.uuid4())[:8]
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        import time
        
        # Process with Docling
        print(f"[{doc_id}] Initializing converter...")
        converter = get_converter()
        
        print(f"[{doc_id}] Starting PDF conversion (this may take a while)...")
        start = time.time()
        result = converter.convert(tmp_path)
        print(f"[{doc_id}] PDF conversion complete ({time.time() - start:.1f}s)")
        
        doc = result.document
        
        return process_document(doc, doc_id, include_images_base64)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
    finally:
        # Clean up temp file
        os.unlink(tmp_path)


@app.post("/extract-from-path", response_model=ExtractionResult)
async def extract_pdf_from_path(
    file_path: str = Form(...),
    include_images_base64: bool = Form(default=False),
    document_id: Optional[str] = Form(default=None)
):
    """
    Extract content from a PDF file at a given path.
    Useful when the PDF is already on the server filesystem.
    
    - **file_path**: Absolute path to the PDF file
    - **include_images_base64**: If true, include base64 encoded image data in response  
    - **document_id**: Optional document ID (will be generated if not provided)
    """
    path = Path(file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    if not path.suffix.lower() == '.pdf':
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Generate document ID if not provided
    doc_id = document_id or hashlib.md5(file_path.encode()).hexdigest()[:8]
    
    try:
        import time
        
        # Process with Docling
        print(f"[{doc_id}] Initializing converter...")
        converter = get_converter()
        
        print(f"[{doc_id}] Starting PDF conversion (this may take a while)...")
        start = time.time()
        result = converter.convert(str(path))
        print(f"[{doc_id}] PDF conversion complete ({time.time() - start:.1f}s)")
        
        doc = result.document
        
        return process_document(doc, doc_id, include_images_base64)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@app.get("/images/{image_id}")
async def get_image(image_id: str):
    """Retrieve a previously extracted image by ID"""
    image_path = IMAGES_DIR / f"{image_id}.png"
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    
    return JSONResponse({
        "id": image_id,
        "base64_data": data,
        "mime_type": "image/png"
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
