import os
from typing import List, Optional
from pydantic import BaseModel, Field

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chat_models import init_chat_model

from langchain_community.document_loaders import PyPDFLoader
from docx import Document
from PIL import Image
import pytesseract

# Load API key
api_key = os.environ["OPENAI_API_KEY"]

# Resume schema
class Edu(BaseModel):
    University: str
    Degree: str
    Gpax: Optional[float] = Field(default=None, ge=0, le=10.0)
    Graduation: Optional[int] = Field(default=None)

class Exp(BaseModel):
    Company: Optional[str] = None
    Duration: Optional[str] = None
    Position: Optional[str] = None
    Responsibilities: Optional[List[str]] = None

class Resume(BaseModel):
    Name: str
    Gender: Optional[str] = None
    DOB: Optional[str] = None
    Age: Optional[int] = None
    Email: str
    Phone: str
    Education: Optional[List[Edu]] = None
    Experience: Optional[List[Exp]] = None
    Skills: Optional[List[str]] = None

# LangChain setup
resume_template = """
You are an AI assistant tasked with extracting structured information from a technical resume.
Only extract fields available in the Resume class below.

Resume Detail:
{resume_text}
"""

parser = PydanticOutputParser(pydantic_object=Resume)
prompt_template = PromptTemplate(template=resume_template, input_variables=["resume_text"])
model = init_chat_model(model="gpt-4o-mini", model_provider="openai").with_structured_output(Resume)

# Format-specific extractors
def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_text_from_image(file_path: str) -> str:
    image = Image.open(file_path)
    return pytesseract.image_to_string(image)

def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        return "\n".join([doc.page_content for doc in docs])
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        return extract_text_from_image(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def parse_resume(file_path: str) -> Resume:
    resume_text = extract_text(file_path)
    prompt = prompt_template.invoke({"resume_text": resume_text})
    result = model.invoke(prompt)
    return result, resume_text
