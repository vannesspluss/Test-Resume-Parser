import os
import signal
import io
import hashlib
from typing import List, Optional
from pydantic import BaseModel, Field, EmailStr, constr, confloat
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from docx import Document
from PIL import Image
import pytesseract

api_key = os.environ.get("OPENAI_API_KEY")
ocr_cache = {}

class PersonalInformation(BaseModel):
    firstNameEN: str
    lastNameEN: str
    firstNameTH: str
    lastNameTH: str
    birthDate: Optional[str] = None
    age: Optional[int] = Field(default=None, ge=0)
    gender: Optional[str] = Field(default=None, pattern="^(Male|Female|Other)$")
    phone: constr(min_length=10, max_length=12)
    email: EmailStr
    province: Optional[str] = None
    district: Optional[str] = None

class Salary(BaseModel):
    lastedSalary: Optional[confloat(ge=0)] = None
    expectSalary: Optional[confloat(ge=0)] = None

class Qualification(BaseModel):
    industry: Optional[str] = None
    experiencesYear: Optional[int] = Field(default=None, ge=0)
    majorSkill: Optional[str] = None
    minorSkill: Optional[str] = None

class Certificate(BaseModel):
    course: Optional[str] = None
    year: Optional[str] = None
    institute: Optional[str] = None

class Experience(BaseModel):
    company: Optional[str] = None
    position: Optional[str] = None
    project: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    responsibility: Optional[str] = None

class Education(BaseModel):
    degreeLevel: str
    program: str
    major: str
    year: str
    university: str

class Resume(BaseModel):
    personalInformation: Optional[PersonalInformation] = None
    availability: Optional[str] = None
    currentPosition: Optional[str] = None
    salary: Optional[Salary] = None
    qualification: Optional[List[Qualification]] = None
    softSkills: Optional[List[str]] = None
    technicalSkills: Optional[List[str]] = None
    experiences: Optional[List[Experience]] = None
    educations: Optional[List[Education]] = None
    certificates: Optional[List[Certificate]] = None


# === Prompt ===
resume_template = """
You are an AI assistant tasked with extracting structured information from a resume.
Extract as much relevant information as possible using the following schema.

Only return the output in the following JSON format for the Resume class.

Resume Detail:
{resume_text}
"""

parser = PydanticOutputParser(pydantic_object=Resume)
prompt_template = PromptTemplate(template=resume_template, input_variables=["resume_text"])
model = init_chat_model(model="gpt-4o-mini", model_provider="openai").with_structured_output(Resume)

# === Extractors ===
def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

class TimeoutException(Exception): pass

def timeout_handler(signum, frame):
    raise TimeoutException()

def compress_image_to_under_100kb(image: Image.Image) -> Image.Image:
    quality = 85
    while quality > 10:
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality, optimize=True)
        size_kb = buffer.tell() / 1024
        if size_kb <= 100:
            buffer.seek(0)
            return Image.open(buffer)
        quality -= 5
    raise Exception("Cannot compress image under 100 KB without losing quality.")

def get_image_hash_from_file(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def extract_text_from_image(file_path: str) -> str:
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(180)

        image_hash = get_image_hash_from_file(file_path)

        if image_hash in ocr_cache:
            return f"[CACHE HIT]\n{ocr_cache[image_hash]}"

        image = Image.open(file_path).convert("L")

        MAX_WIDTH, MAX_HEIGHT = 250, 300
        width_ratio = MAX_WIDTH / image.width
        height_ratio = MAX_HEIGHT / image.height
        scale_ratio = min(width_ratio, height_ratio)
        if scale_ratio < 1.0:
            new_size = (int(image.width * scale_ratio), int(image.height * scale_ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        image = image.point(lambda x: 0 if x < 140 else 255, '1')

        if os.path.getsize(file_path) > 100 * 1024:
            image = compress_image_to_under_100kb(image)

        config = "--oem 3 --psm 6 -l eng+tha"
        text = pytesseract.image_to_string(image, config=config)

        ocr_cache[image_hash] = text
        print("OCR Extracted Text:", text)
        return f"[OCR DONE]\n{text}"

    except TimeoutException:
        return "OCR timed out. Try a smaller or clearer image."
    except Exception as e:
        return f"OCR failed: {str(e)}"
    finally:
        signal.alarm(0)

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

    # CLEAN OCR header --> fix image parser
    if resume_text.startswith("[OCR DONE]") or resume_text.startswith("[CACHE HIT]"):
        resume_text = resume_text.split("]", 1)[-1].strip()
    
    prompt = prompt_template.invoke({"resume_text": resume_text})
    result = model.invoke(prompt)
    return result, resume_text
