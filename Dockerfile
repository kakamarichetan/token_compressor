FROM python:3.10-slim

# Install system dependencies for building native extensions
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libmariadb-dev \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install required Python packages (including streamlit, en_core_web_sm, etc.)
RUN pip install --upgrade pip
RUN pip install streamlit spacy
RUN python -m spacy download en_core_web_sm

# Set up working directory
WORKDIR /app

# Copy your app files
COPY . .

# Install app dependencies
RUN pip install -r requirements.txt

# Expose the Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py"]
