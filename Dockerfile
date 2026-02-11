FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY similarity_service_enhanced.py .
COPY crime_reports_cleaned.csv .
COPY description_embeddings.npy .
COPY app.py .

# Expose Hugging Face default port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
