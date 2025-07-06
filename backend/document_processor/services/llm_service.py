"""
LLM Service for entity extraction from documents.
Uses OpenAI GPT models for structured data extraction.
"""

import os
import json
import openai
import logging
from django.conf import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Service for Large Language Model operations using OpenAI."""
    
    def __init__(self):
        """Initialize LLM service with OpenAI configuration."""
        # Set OpenAI API key
        api_key = getattr(settings, 'OPENAI_API_KEY', None) or os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.warning("OpenAI API key not found. LLM functionality will be limited.")
            self.client = None
        else:
            try:
                openai.api_key = api_key
                self.client = openai
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {e}")
                self.client = None
        
        # Document type specific field mappings
        self.document_fields = {
            'invoice': [
                'invoice_number', 'date', 'due_date', 'vendor_name', 
                'customer_name', 'total_amount', 'tax_amount', 'subtotal',
                'billing_address', 'shipping_address', 'payment_terms'
            ],
            'receipt': [
                'merchant_name', 'date', 'time', 'total_amount', 'tax_amount',
                'items', 'payment_method', 'transaction_id', 'cashier'
            ],
            'contract': [
                'contract_title', 'parties', 'effective_date', 'expiration_date',
                'contract_value', 'governing_law', 'signature_required',
                'termination_clause', 'renewal_terms'
            ],
            'id_document': [
                'document_type', 'full_name', 'date_of_birth', 'document_number',
                'issue_date', 'expiration_date', 'issuing_authority', 
                'address', 'nationality', 'sex'
            ],
            'bank_statement': [
                'account_number', 'account_holder', 'statement_period',
                'opening_balance', 'closing_balance', 'transactions',
                'bank_name', 'routing_number'
            ],
            'medical_record': [
                'patient_name', 'patient_id', 'date_of_birth', 'date_of_visit',
                'doctor_name', 'diagnosis', 'treatment', 'medications',
                'allergies', 'vital_signs'
            ]
        }
    
    def extract_entities(self, text, document_type):
        """
        Extract structured entities from document text.
        
        Args:
            text (str): Document text
            document_type (str): Type of document
            
        Returns:
            dict: Extracted entities or None if extraction fails
        """
        try:
            if not self.client:
                logger.error("OpenAI client not available")
                return None
            
            # Get fields for this document type
            fields = self.document_fields.get(document_type, [])
            if not fields:
                logger.warning(f"No field mapping for document type: {document_type}")
                fields = ['key_information', 'important_details']
            
            # Create prompt
            prompt = self._create_extraction_prompt(text, document_type, fields)
            
            # Call OpenAI API
            response = self._call_openai_api(prompt)
            
            if response:
                # Parse and validate response
                entities = self._parse_llm_response(response, document_type)
                return entities
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return None
    
    def _create_extraction_prompt(self, text, document_type, fields):
        """Create extraction prompt for the LLM."""
        field_list = ', '.join(fields)
        
        prompt = f"""Parse and extract information from the following {document_type} document.
Extract these specific fields: {field_list}.

Instructions:
1. Return ONLY a valid JSON object with the extracted information
2. Use null for fields that cannot be found or determined
3. For dates, use YYYY-MM-DD format when possible
4. For amounts, include currency symbol if present
5. Be accurate and conservative - only extract information you're confident about
6. If a field contains multiple items (like transaction list), format as an array

Document Text:
{text}

JSON Response:"""
        
        return prompt
    
    def _call_openai_api(self, prompt, model="gpt-3.5-turbo", max_tokens=1000):
        """Call OpenAI API with the given prompt."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert document analysis assistant. Extract structured information from documents and return valid JSON only."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.1,  # Low temperature for more consistent results
                response_format={"type": "json_object"}  # Ensure JSON response
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return None
    
    def _parse_llm_response(self, response, document_type):
        """Parse and validate LLM response."""
        try:
            # Try to parse JSON
            entities = json.loads(response)
            
            # Validate and clean the response
            cleaned_entities = self._clean_entities(entities, document_type)
            
            return cleaned_entities
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from LLM: {e}")
            # Try to extract JSON from response if it's wrapped in text
            return self._extract_json_from_text(response)
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return None
    
    def _extract_json_from_text(self, text):
        """Try to extract JSON from text that might contain additional content."""
        try:
            # Look for JSON object in the text
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx:end_idx + 1]
                return json.loads(json_str)
            
            return None
            
        except Exception as e:
            logger.error(f"Could not extract JSON from text: {e}")
            return None
    
    def _clean_entities(self, entities, document_type):
        """Clean and validate extracted entities."""
        if not isinstance(entities, dict):
            return {}
        
        cleaned = {}
        expected_fields = self.document_fields.get(document_type, [])
        
        for key, value in entities.items():
            # Convert key to lowercase and remove spaces
            clean_key = key.lower().replace(' ', '_').replace('-', '_')
            
            # Skip empty or null values
            if value is None or value == "" or value == "null":
                continue
            
            # Clean string values
            if isinstance(value, str):
                value = value.strip()
                if value.lower() in ['null', 'none', 'n/a', 'not found', '']:
                    continue
            
            cleaned[clean_key] = value
        
        return cleaned
    
    def get_confidence_score(self, text, extracted_entities):
        """
        Calculate confidence score for extracted entities.
        
        Args:
            text (str): Original document text
            extracted_entities (dict): Extracted entities
            
        Returns:
            float: Confidence score between 0 and 1
        """
        try:
            if not extracted_entities:
                return 0.0
            
            # Count non-empty fields
            filled_fields = sum(1 for v in extracted_entities.values() if v)
            total_possible_fields = len(extracted_entities)
            
            if total_possible_fields == 0:
                return 0.0
            
            # Base confidence on filled fields ratio
            field_confidence = filled_fields / total_possible_fields
            
            # Adjust based on text length (longer text usually means better extraction)
            text_length_factor = min(1.0, len(text) / 1000)  # Normalize to 1000 chars
            
            # Combine factors
            confidence = (field_confidence * 0.7) + (text_length_factor * 0.3)
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.0
    
    def is_available(self):
        """Check if LLM service is available."""
        return self.client is not None
    
    def get_supported_document_types(self):
        """Get list of supported document types."""
        return list(self.document_fields.keys())
    
    def get_fields_for_document_type(self, document_type):
        """Get expected fields for a document type."""
        return self.document_fields.get(document_type, [])
