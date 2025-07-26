FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Provide an entrypoint for running the bot.  The configuration file
# should be mounted into the container at `/app/config.yaml`.  When
# running locally you can override the command via the docker run
# command line.
CMD ["python", "main.py", "--config", "config.yaml"]