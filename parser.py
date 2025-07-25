import os
import re
import signal
from typing import List, Optional
from pydantic import BaseModel, Field, EmailStr, constr, confloat

import pytesseract
import easyocr
from PIL import Image
from docx import Document
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader

# Pydantic Models
class PersonalInformation(BaseModel):
    firstNameEN: str
    lastNameEN: str
    firstNameTH: str
    lastNameTH: str
    birthDate: Optional[str]
    age: Optional[int]
    gender: Optional[str] = Field(default=None, pattern="^(Male|Female|Other)$")
    phone: constr(min_length=10, max_length=12)
    email: EmailStr
    province: Optional[str]
    district: Optional[str]

class Salary(BaseModel):
    lastedSalary: Optional[confloat(ge=0)]
    expectSalary: Optional[confloat(ge=0)]

class Qualification(BaseModel):
    industry: Optional[str]
    experiencesYear: Optional[int]
    majorSkill: Optional[str]
    minorSkill: Optional[str]

class Certificate(BaseModel):
    course: Optional[str]
    year: Optional[str]
    institute: Optional[str]

class Experience(BaseModel):
    company: Optional[str]
    position: Optional[str]
    project: Optional[str]
    startDate: Optional[str]
    endDate: Optional[str]
    responsibility: Optional[str]

class Education(BaseModel):
    degreeLevel: str
    program: str
    major: str
    year: str
    university: str

class Resume(BaseModel):
    personalInformation: Optional[PersonalInformation]
    availability: Optional[str]
    currentPosition: Optional[str]
    salary: Optional[Salary]
    qualification: Optional[List[Qualification]]
    softSkills: Optional[List[str]]
    technicalSkills: Optional[List[str]]
    experiences: Optional[List[Experience]]
    educations: Optional[List[Education]]
    certificates: Optional[List[Certificate]]

resume_template = """
You are an AI assistant tasked with extracting structured information from a technical resume.
Only extract the information that is present in the Resume class.

Resume Detail:
{resume_text}
"""

parser = PydanticOutputParser(pydantic_object=Resume)
prompt_template = PromptTemplate(
    template=resume_template,
    input_variables=['resume_text']
)

model = init_chat_model(model='gpt-4o-mini', model_provider='openai').with_structured_output(Resume)

# OCR and File Extraction
class TimeoutException(Exception): pass

def timeout_handler(signum, frame):
    raise TimeoutException()

def extract_text_from_pdf(file_path: str) -> str:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return "\n".join([doc.page_content for doc in docs])

def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_easyocr_text(file_path: str):
    reader = easyocr.Reader(['en', 'th'], gpu=False)
    results = reader.readtext(file_path)
    text = " ".join([res[1] for res in results])
    avg_conf = sum([res[2] for res in results]) / len(results) if results else 0
    return text, avg_conf

def extract_tesseract_text(file_path: str, timeout=180):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        image = Image.open(file_path)
        custom_config = "--oem 3 --psm 6 -l eng"
        text = pytesseract.image_to_string(image, config=custom_config)
        return text.strip()
    except Exception:
        return ""
    finally:
        signal.alarm(0)

def contains_thai(text: str) -> bool:
    return bool(re.search(r'[\u0E00-\u0E7F]', text))

def count_valid_words(text: str) -> int:
    words = re.findall(r'\b\w+\b', text)
    return len(words)

def extract_text_from_image(file_path: str) -> str:
    easy_text, easy_conf = extract_easyocr_text(file_path)
    if contains_thai(easy_text):
        return easy_text
    tesseract_text = extract_tesseract_text(file_path)
    return tesseract_text if count_valid_words(tesseract_text) > count_valid_words(easy_text) else easy_text

def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
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
