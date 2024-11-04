# Step 1: Use the official Python image with your desired Python version
FROM python:3.11-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the requirements.txt file into the container
COPY requirements.txt /app/requirements.txt

# Step 4: Install the dependencies specified in requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Step 5: Copy the rest of your code into the container
COPY . /app

# Step 6: Expose any ports your app will use (if needed)
# EXPOSE 5000  # Uncomment this if you're serving an API, for example

# Step 7: Define the command to run your code
# Change this to the entry point of your application
CMD ["python", "query_data.py", "What happens when a player lands on Free Parking in Monopoly?"]
