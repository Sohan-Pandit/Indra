
import sys
# Force UTF-8 for stdout/stderr so tracebacks with non-ASCII chars
# (e.g. → in mock records) don't crash the exception handler on Windows.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from src.annotation.llm_labeler import LLMLabeler
import uvicorn
import json
import time
import re

app = FastAPI(title="Indra API", description="Backend for Indra Climate Impact Extractor")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnnotateRequest(BaseModel):
    abstract: str
    pub_year: Optional[str] = ""

class AnnotateResponse(BaseModel):
    record: dict
    uncertaintyAnalysis: str
    secondaryImpacts: Optional[str]
    rawResponse: str



import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.post("/annotate", response_model=AnnotateResponse)
async def annotate(request: AnnotateRequest, x_api_key: str = Header(...)):
    # No API key provided
    if not x_api_key or x_api_key == "PLACEHOLDER_API_KEY" or x_api_key == "test_key":
        raise HTTPException(status_code=401, detail="No API key provided. Please enter a valid key from: Google Gemini (AIza...), Anthropic (sk-ant-...), OpenAI (sk-...), Grok/xAI (xai-...), or Groq (gsk_...).")


    try:
        # Initialize wrapper — auto-detects provider from key prefix
        from src.llm.wrapper import LLMWrapper
        llm = LLMWrapper(api_key=x_api_key)
        logger.info(f"Using provider: {llm.provider} / model: {llm.model_name}")
        
        system_instruction = """
You are Indra, an expert annotator for climate impact literature built for the CLIMPACT-TEXT research pipeline.

When a user pastes a scientific abstract, extract a structured impact record and return it as valid JSON following this exact schema:

{
  "abstract_id": "USER_INPUT",
  "hazard_type": "<flood | drought | heatwave | extreme_rainfall | tropical_cyclone | wildfire | storm | landslide | sea_level_rise | other>",
  "hazard_intensity": "<quantified intensity if mentioned, else null>",
  "location": {
    "raw": "<location as written in abstract>",
    "normalized": "<City, COUNTRY_CODE>"
  },
  "time_period": {
    "raw": "<time expression as written>",
    "normalized": "<ISO 8601 interval: YYYY-MM-DD/YYYY-MM-DD>"
  },
  "impact_domain": "<mortality | morbidity | displacement | economic_loss | infrastructure | agriculture | ecosystem | mental_health | food_security | water_security | other>",
  "impact_type": "<observed | projected | modeled | unknown>",
  "impact_magnitude": "<quantified impact if mentioned, else null>",
  "magnitude_vague": "<true | false>",
  "affected_group": "<specific group if mentioned, else null>",
  "causal_relation": {
    "subject": "<hazard entity>",
    "predicate": "<caused | contributed_to | associated_with | mitigated | none>",
    "object": "<impact entity>",
    "dependency_path": "<syntactic path if identifiable>"
  },
  "uncertainty_level": "<low | medium | high | unknown>",
  "uncertainty_source": "<observational | modeled | survey | expert_judgment | null>",
  "hedge_terms": ["<list of hedging words found in abstract>"],
  "grounding_quote": "<exact verbatim quote from abstract supporting the main impact claim>",
  "grounding_verified": true
}

After the JSON, provide a brief section called UNCERTAINTY ANALYSIS:
- List every hedge term found and classify it as high / medium / low uncertainty signal
- State whether the impact is observed or projected and why
- Flag any fields where the abstract was ambiguous or information was missing

If there are secondary impacts, list them in a section called SECONDARY IMPACTS.

Rules:
- Never hallucinate. If a field is not supported by the abstract text, set it to null.
- grounding_quote must be an exact verbatim substring of the abstract.
- Return only one impact record per abstract.
"""
        user_content = f"Abstract:\n{request.abstract}\n\nPublication Year (if known): {request.pub_year}"
        
        raw_text = llm.generate(user_content, system_instruction=system_instruction)
        
        # Parse logic (mirrors frontend)
        json_record = {}
        uncertainty_analysis = "No analysis provided."
        secondary_impacts = None
        
        json_start = raw_text.find('{')
        json_end = raw_text.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            try:
                json_record = json.loads(raw_text[json_start:json_end])
                # Validate with our Schema!
                from src.schema.impact_schema import validate_record
                is_valid, errors = validate_record(json_record)
                if not is_valid:
                    json_record["validation_errors"] = errors
            except Exception:
                pass
                
        analysis_match = re.search(r"UNCERTAINTY ANALYSIS:([\s\S]*?)(?=SECONDARY IMPACTS|$)", raw_text, re.IGNORECASE)
        if analysis_match:
            uncertainty_analysis = analysis_match.group(1).strip()
            
        secondary_match = re.search(r"SECONDARY IMPACTS:([\s\S]*?)$", raw_text, re.IGNORECASE)
        if secondary_match:
            secondary_impacts = secondary_match.group(1).strip()
            
        return AnnotateResponse(
            record=json_record,
            uncertaintyAnalysis=uncertainty_analysis,
            secondaryImpacts=secondary_impacts,
            rawResponse=raw_text
        )

    except Exception as e:
        logger.exception("Annotation failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
