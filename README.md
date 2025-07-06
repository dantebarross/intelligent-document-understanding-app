# Intelligent Document Understanding API

A complete end-to-end document processing API that extracts structured information from unstructured documents using OCR technology, vector database retrieval, and large language models.

## Overview

This project implements an intelligent document understanding system that can:

- **Accept document uploads** (PDF and image formats)
- **Extract text content** using Optical Character Recognition (OCR)
- **Identify document type** via semantic search against a vector database
- **Extract structured data** using Large Language Models (LLMs)
- **Return standardized JSON** containing all extracted information

## Architecture

```
intelligent-document-understanding-api/
├── backend/                    # Django REST API
│   ├── document_processor/     # Main application
│   ├── intelligent_document_api/  # Django project settings
│   ├── requirements.txt        # Python dependencies
│   ├── Dockerfile             # Backend containerization
│   └── init_vector_db.py      # Vector DB initialization script
├── frontend/                   # React web interface
│   ├── src/                   # React source code
│   ├── public/                # Static assets
│   ├── package.json           # Node.js dependencies
│   └── Dockerfile             # Frontend containerization
├── examples/                   # API usage examples
├── docker-compose.yml         # Multi-service orchestration
├── setup.py                   # Automated setup script
└── README.md                  # This file
```

## Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd intelligent-document-understanding-api

# Run the automated setup script
python setup.py
```

### Option 2: Manual Setup

#### Prerequisites

- Python 3.8+
- Node.js 16+
- Tesseract OCR
  - **Windows**: Install from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki) (recommended)
  - **Linux**: `sudo apt-get install tesseract-ocr`
  - **Mac**: `brew install tesseract`
- OpenAI API Key

#### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
# Edit .env and add your OpenAI API key

# Run migrations
python manage.py migrate

# Create superuser (optional - for Django admin access)
python manage.py createsuperuser
# Follow prompts:
# - Username: admin (or any username)
# - Email: (optional, can press Enter)
# - Password: min 8 chars, not too common (e.g., admin123456)

# Start development server
python manage.py runserver
```

#### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

### Option 3: Docker Setup

```bash
# Start all services
docker-compose up --build

# Access the application at:
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
```

## Vector Database Initialization

The system uses the Real World Documents Collections dataset for training and testing:

```bash
# Quick initialization with sample documents (5 per category)
cd backend
python init_vector_db.py

# Full training dataset processing (all categories)
python init_vector_db_large.py
```

The dataset is organized as:
- **training_data/**: Training documents (70% of dataset)
- **testing_data/**: Testing documents (30% of dataset)  
- **media/**: Django upload folder for API usage

### Training Dataset

For advanced training and testing, this project uses the **Real World Documents Collections** dataset from Kaggle:

**Dataset**: [Real World Documents Collections](https://www.kaggle.com/datasets/shaz13/real-world-documents-collections)

This dataset contains diverse document types and formats which can be used to:
- Train the document classification model
- Test OCR accuracy on various document qualities
- Benchmark the system's performance

To use the full training dataset:
1. Download the dataset from Kaggle
2. Extract to `backend/training_data/`
3. Run `python init_vector_db_large.py` to process the full dataset

## Configuration

### Backend Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# Django Configuration
SECRET_KEY=your-secret-key-here
DEBUG=True

# OpenAI API Configuration (required)
OPENAI_API_KEY=your-openai-api-key-here

# OCR Configuration
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe  # Windows
# TESSERACT_CMD=/usr/bin/tesseract  # Linux/Mac

# Vector Database Settings
VECTOR_DB_TYPE=faiss
EMBEDDING_MODEL=all-MiniLM-L6-v2

# API Configuration
ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

### Frontend Environment Variables

Create a `.env` file in the `frontend/` directory:

```env
REACT_APP_API_URL=http://localhost:8000
GENERATE_SOURCEMAP=false
```

## API Documentation

### Extract Entities Endpoint

**POST** `/extract_entities/`

Upload a document and extract structured information.

#### Request

```bash
curl -X POST \
  http://localhost:8000/extract_entities/ \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/document.pdf'
```

#### Response

```json
{
  "success": true,
  "document_type": "invoice",
  "confidence": 0.92,
  "entities": {
    "invoice_number": "INV-12345",
    "date": "2024-01-01",
    "total_amount": "$450.00",
    "vendor_name": "ABC Corp"
  },
  "extracted_text": "INVOICE\nInvoice Number: INV-12345...",
  "processing_time": "1.25s"
}
```

### Supported Document Types

- **Invoice**: Business invoices with vendor, customer, amounts, dates
- **Receipt**: Purchase receipts with merchant, items, totals  
- **Contract**: Legal agreements with parties, dates, terms
- **ID Document**: Identity documents like driver's licenses, passports
- **Bank Statement**: Financial statements with transactions, balances
- **Medical Record**: Healthcare documents with patient info, treatments

### Supported File Formats

- PDF (.pdf)
- Images (.png, .jpg, .jpeg)
- Maximum file size: 10 MB

## Testing

### Running Tests

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

### Test Coverage

```bash
# Generate coverage report
cd backend
pytest --cov=document_processor --cov-report=html
```

## Web Interface

The React frontend provides an intuitive interface for:

- **File Upload**: Drag & drop or click to upload documents
- **Progress Tracking**: Real-time processing status
- **Results Display**: Formatted JSON output with extracted entities
- **Error Handling**: Clear error messages and retry options

Access the web interface at: `http://localhost:3000`

## Architecture Details

### Backend Components

- **OCR Service**: Text extraction using Tesseract and EasyOCR
- **Vector Database**: Document classification using FAISS embeddings
- **LLM Integration**: Entity extraction using OpenAI GPT models
- **REST API**: Django REST Framework endpoints
- **File Handling**: Secure upload and processing pipeline

### Frontend Components

- **File Upload**: React dropzone with validation
- **API Integration**: Axios HTTP client with error handling
- **UI Components**: Material-UI for consistent design
- **State Management**: React hooks for application state

## Performance

### Typical Processing Times

- Simple documents (1 page): 0.5-1.5 seconds
- Complex documents (multi-page): 1.5-3.0 seconds
- Large images: 2.0-4.0 seconds

### Optimization Features

- Asynchronous processing
- Image preprocessing for better OCR
- Confidence scoring for reliability
- Caching for repeated requests

## Security

- File type validation
- File size limits
- Input sanitization
- CORS configuration
- Environment variable protection

## Deployment

### Production Considerations

1. **Environment Variables**: Set production values in `.env`
2. **Database**: Configure PostgreSQL for production
3. **Static Files**: Configure static file serving
4. **HTTPS**: Enable SSL/TLS certificates
5. **Monitoring**: Add logging and monitoring tools

### Docker Production

```bash
# Build production images
docker-compose -f docker-compose.prod.yml up --build
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Examples

See the `examples/` directory for:

- Complete API request/response examples
- Code samples in Python and JavaScript
- Error handling scenarios
- Integration examples

## Troubleshooting

### Common Issues

1. **OCR Not Working**: Install Tesseract OCR and verify path in `.env`
2. **OpenAI API Errors**: Check API key and billing status
3. **Memory Issues**: Reduce file sizes or increase system memory
4. **CORS Errors**: Verify frontend URL in backend CORS settings

### Debug Mode

Enable debug logging by setting `DEBUG=True` in backend `.env` file.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Tesseract OCR for text extraction
- OpenAI for language model capabilities
- FAISS for efficient vector search
- Django REST Framework for API development
- React for the frontend interface

---

**Need help?** Check the `examples/` directory or open an issue on GitHub.
