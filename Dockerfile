# Dockerfile for Streamlit
FROM python:3.9  # or another Python version as needed

# Set the working directory
WORKDIR ./

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application to the container
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]
