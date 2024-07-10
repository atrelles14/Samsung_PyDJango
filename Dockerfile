# Use the official Python image as a parent image
FROM python:3.11.5

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY .   /app/

# Expose the port the app runs on
EXPOSE 8000

# Set environment variables for Django
ENV DJANGO_SETTINGS_MODULE=apiData.settings
ENV PYTHONUNBUFFERED 1

# Run the Django development server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
