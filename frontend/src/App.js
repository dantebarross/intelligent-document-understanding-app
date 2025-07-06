import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState('idle'); // idle, processing, success, error

  // File validation
  const validateFile = (file) => {
    const allowedTypes = ['application/pdf', 'image/jpeg', 'image/jpg', 'image/png'];
    const maxSize = 10 * 1024 * 1024; // 10MB

    if (!allowedTypes.includes(file.type)) {
      return 'Only PDF, JPG, JPEG, and PNG files are supported';
    }

    if (file.size > maxSize) {
      return 'File size must be less than 10MB';
    }

    return null;
  };

  // Handle file drop
  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    setError(null);
    setResults(null);
    setStatus('idle');

    if (rejectedFiles.length > 0) {
      setError('Invalid file format. Please upload PDF, JPG, JPEG, or PNG files only.');
      return;
    }

    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      const validationError = validateFile(file);
      
      if (validationError) {
        setError(validationError);
        return;
      }

      setSelectedFile(file);
    }
  }, []);

  // Configure dropzone
  const { getRootProps, getInputProps, isDragActive, isDragAccept, isDragReject } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/png': ['.png']
    },
    multiple: false,
    maxSize: 10 * 1024 * 1024 // 10MB
  });

  // Upload and process file
  const handleUpload = async () => {
    if (!selectedFile) return;

    setUploading(true);
    setStatus('processing');
    setError(null);
    setProgress(0);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_BASE_URL}/extract_entities/`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setProgress(percentCompleted);
        },
      });

      if (response.data.success) {
        setResults(response.data);
        setStatus('success');
      } else {
        setError(response.data.error || 'Processing failed');
        setStatus('error');
      }
    } catch (error) {
      console.error('Upload error:', error);
      if (error.response?.data?.error) {
        setError(error.response.data.error);
      } else if (error.response?.status === 413) {
        setError('File too large. Maximum size is 10MB.');
      } else if (error.code === 'NETWORK_ERROR') {
        setError('Cannot connect to server. Please ensure the backend is running.');
      } else {
        setError('An unexpected error occurred. Please try again.');
      }
      setStatus('error');
    } finally {
      setUploading(false);
      setProgress(0);
    }
  };

  // Reset form
  const handleReset = () => {
    setSelectedFile(null);
    setResults(null);
    setError(null);
    setStatus('idle');
    setProgress(0);
  };

  // Format file size
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // Get dropzone class names
  const getDropzoneClassName = () => {
    let className = 'dropzone';
    if (isDragActive) className += ' active';
    if (isDragAccept) className += ' accept';
    if (isDragReject) className += ' reject';
    return className;
  };

  return (
    <div className="container">
      <header className="header">
        <h1>🤖 Intelligent Document Understanding</h1>
        <p>Extract structured information from your documents using AI</p>
      </header>

      <div className="main-content">
        {/* Upload Section */}
        <div className="card upload-section">
          <h2>📄 Upload Document</h2>
          
          <div {...getRootProps()} className={getDropzoneClassName()}>
            <input {...getInputProps()} />
            <div className="upload-icon">📁</div>
            {isDragActive ? (
              <p>Drop the file here...</p>
            ) : (
              <>
                <p><strong>Drag & drop a document here</strong></p>
                <p>or click to select a file</p>
                <p style={{ fontSize: '0.9rem', color: '#666', marginTop: '10px' }}>
                  Supported: PDF, JPG, JPEG, PNG (max 10MB)
                </p>
              </>
            )}
          </div>

          {selectedFile && (
            <div className="file-info">
              <h4>📋 Selected File</h4>
              <p><strong>Name:</strong> {selectedFile.name}</p>
              <p><strong>Size:</strong> {formatFileSize(selectedFile.size)}</p>
              <p><strong>Type:</strong> {selectedFile.type}</p>
            </div>
          )}

          {uploading && (
            <div className="progress-container">
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{ width: `${progress}%` }}
                ></div>
              </div>
              <div className="status-message status-processing">
                {progress < 100 ? (
                  <>
                    <span className="spinner"></span>
                    Uploading... {progress}%
                  </>
                ) : (
                  <>
                    <span className="spinner"></span>
                    Processing document...
                  </>
                )}
              </div>
            </div>
          )}

          {status === 'success' && (
            <div className="status-message status-success">
              ✅ Document processed successfully!
            </div>
          )}

          {error && (
            <div className="error-message">
              ❌ {error}
            </div>
          )}

          <div style={{ display: 'flex', gap: '10px', marginTop: '20px' }}>
            <button 
              className="btn" 
              onClick={handleUpload}
              disabled={!selectedFile || uploading}
              style={{ flex: 1 }}
            >
              {uploading ? (
                <>
                  <span className="spinner"></span>
                  Processing...
                </>
              ) : (
                '🚀 Extract Entities'
              )}
            </button>
            
            {(selectedFile || results || error) && (
              <button 
                className="btn" 
                onClick={handleReset}
                disabled={uploading}
                style={{ 
                  flex: 0, 
                  background: '#6c757d',
                  minWidth: '100px'
                }}
              >
                🔄 Reset
              </button>
            )}
          </div>
        </div>

        {/* Results Section */}
        <div className="card results-section">
          <h2>📊 Extraction Results</h2>
          
          {!results && !error && (
            <div className="empty-state">
              <div className="empty-state-icon">📝</div>
              <p>Upload a document to see extracted information here</p>
            </div>
          )}

          {results && (
            <div className="results-container">
              <div>
                <span className="document-type">
                  📄 {results.document_type.toUpperCase()}
                </span>
                <span className="confidence-score">
                  Confidence: {(results.confidence * 100).toFixed(1)}%
                </span>
              </div>

              <h3>🎯 Extracted Entities</h3>
              <div className="json-display">
                {JSON.stringify(results.entities, null, 2)}
              </div>

              {results.extracted_text && (
                <>
                  <h3 style={{ marginTop: '25px' }}>📝 Extracted Text (Preview)</h3>
                  <div className="json-display" style={{ maxHeight: '200px' }}>
                    {results.extracted_text.substring(0, 1000)}
                    {results.extracted_text.length > 1000 && '...'}
                  </div>
                </>
              )}

              {results.processing_time && (
                <div className="processing-time">
                  ⏱️ Processing time: {results.processing_time}
                </div>
              )}

              <h3 style={{ marginTop: '25px' }}>🔍 Full Response</h3>
              <div className="json-display">
                {JSON.stringify(results, null, 2)}
              </div>
            </div>
          )}
        </div>
      </div>

      <footer style={{ 
        textAlign: 'center', 
        marginTop: '40px', 
        padding: '20px',
        color: 'rgba(255,255,255,0.8)',
        fontSize: '0.9rem'
      }}>
        <p>🔬 Powered by OCR, Vector Search & Large Language Models</p>
        <p>Supports: Invoice • Receipt • Contract • ID Document • Bank Statement • Medical Record</p>
      </footer>
    </div>
  );
}

export default App;
