from flask import Flask, render_template, request
import subprocess
import tempfile
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    output = ""
    if request.method == "POST":
        user_text = request.form["user_text"]

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt") as temp_file:
            temp_file.write(user_text)
            temp_file_path = temp_file.name

        try:
            # Run shell script on the file
            result = subprocess.run(["bash", "script.sh", temp_file_path], capture_output=True, text=True)
            output = result.stdout + result.stderr
        finally:
            # Clean up the temp file
            os.remove(temp_file_path)

    return render_template("index.html", output=output)

if __name__ == "__main__":
    app.run(debug=True)
