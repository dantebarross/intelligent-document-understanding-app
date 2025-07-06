#!/usr/bin/env python3
"""
Setup script for Intelligent Document Understanding API
This script automates the initial setup process for both backend and frontend.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description, check=True):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"Running: {command}")
    
    try:
        if platform.system() == "Windows":
            result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        else:
            result = subprocess.run(command.split(), check=check, capture_output=True, text=True)
        
        if result.stdout:
            print(result.stdout)
        if result.stderr and check:
            print(f"Warning: {result.stderr}")
        
        print(f"✅ {description} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}")
        print(f"Command failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error in {description}: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_node_version():
    """Check if Node.js is installed."""
    print("🔍 Checking Node.js installation...")
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True, check=True)
        print(f"✅ Node.js {result.stdout.strip()} found")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Node.js not found. Please install Node.js 16+ from https://nodejs.org/")
        return False

def setup_backend():
    """Set up the backend environment."""
    print("\n🚀 Setting up Backend...")
    
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("❌ Backend directory not found")
        return False
    
    os.chdir(backend_dir)
    
    # Create virtual environment
    venv_path = "venv"
    if not Path(venv_path).exists():
        if not run_command("python -m venv venv", "Creating virtual environment"):
            return False
    
    # Determine activation script based on OS
    if platform.system() == "Windows":
        activate_script = f"{venv_path}\\Scripts\\activate"
        pip_cmd = f"{venv_path}\\Scripts\\pip"
        python_cmd = f"{venv_path}\\Scripts\\python"
    else:
        activate_script = f"{venv_path}/bin/activate"
        pip_cmd = f"{venv_path}/bin/pip"
        python_cmd = f"{venv_path}/bin/python"
    
    # Install requirements
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing Python dependencies"):
        return False
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        print("📝 Creating .env file...")
        env_content = """# Django Configuration
SECRET_KEY=dev-secret-key-change-in-production
DEBUG=True

# OpenAI API Configuration (required for LLM functionality)
OPENAI_API_KEY=your-openai-api-key-here

# OCR Configuration
TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe

# Vector Database Settings
VECTOR_DB_TYPE=faiss
EMBEDDING_MODEL=all-MiniLM-L6-v2

# API Configuration
ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
"""
        env_file.write_text(env_content)
        print("✅ .env file created")
        print("⚠️  Please update OPENAI_API_KEY in backend/.env file")
    
    # Run migrations
    if not run_command(f"{python_cmd} manage.py migrate", "Running database migrations"):
        return False
    
    # Create superuser (optional)
    print("\n📝 You can create a Django superuser now (optional):")
    print("This will allow access to Django admin at http://localhost:8000/admin/")
    create_superuser = input("Create superuser? (y/N): ").lower().strip()
    if create_superuser == 'y':
        print("\n📋 Django superuser creation guide:")
        print("1. Username: Choose any username (e.g., admin)")
        print("2. Email: Optional, can press Enter to skip")
        print("3. Password: Must be at least 8 characters, not too common")
        print("   Example passwords: admin123456, mypassword123, developer2024")
        print("4. Confirm password: Type the same password again")
        print("\nPress Ctrl+C to cancel if needed")
        run_command(f"{python_cmd} manage.py createsuperuser", "Creating superuser", check=False)
    
    os.chdir("..")
    print("✅ Backend setup completed")
    return True

def setup_frontend():
    """Set up the frontend environment."""
    print("\n🚀 Setting up Frontend...")
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return False
    
    os.chdir(frontend_dir)
    
    # Install npm dependencies
    if not run_command("npm install", "Installing Node.js dependencies"):
        return False
    
    # Create .env file for frontend
    env_file = Path(".env")
    if not env_file.exists():
        print("📝 Creating frontend .env file...")
        env_content = """REACT_APP_API_URL=http://localhost:8000
GENERATE_SOURCEMAP=false
"""
        env_file.write_text(env_content)
        print("✅ Frontend .env file created")
    
    os.chdir("..")
    print("✅ Frontend setup completed")
    return True

def create_sample_directories():
    """Create sample document directories."""
    print("\n📁 Creating sample document directories...")
    
    sample_dir = Path("backend/sample_documents")
    document_types = ["invoice", "receipt", "contract", "id_document", "bank_statement", "medical_record"]
    
    for doc_type in document_types:
        type_dir = sample_dir / doc_type
        type_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a README in each directory
        readme_file = type_dir / "README.md"
        readme_content = f"""# {doc_type.replace('_', ' ').title()} Documents

Place sample {doc_type.replace('_', ' ')} documents in this directory.

Supported formats:
- PDF (.pdf)
- Images (.png, .jpg, .jpeg)

These documents will be used to train the document classification system.
"""
        readme_file.write_text(readme_content)
    
    print("✅ Sample document directories created")

def main():
    """Main setup function."""
    print("🎯 Intelligent Document Understanding API Setup")
    print("=" * 50)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_node_version():
        sys.exit(1)
    
    # Setup backend
    if not setup_backend():
        print("❌ Backend setup failed")
        sys.exit(1)
    
    # Setup frontend
    if not setup_frontend():
        print("❌ Frontend setup failed")
        sys.exit(1)
    
    # Create sample directories
    create_sample_directories()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Update backend/.env with your OpenAI API key")
    print("2. Add sample documents to backend/sample_documents/ subdirectories")
    print("3. Initialize vector database: cd backend && python init_vector_db.py")
    print("4. Start backend: cd backend && venv/Scripts/python manage.py runserver (Windows)")
    print("   or: cd backend && venv/bin/python manage.py runserver (Linux/Mac)")
    print("5. Start frontend: cd frontend && npm start")
    print("\n🌐 Access the application at http://localhost:3000")

if __name__ == "__main__":
    main()
