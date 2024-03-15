# Running the Web Application Locally

## Prerequisites
- Ensure you have Python 3.x installed on your system. You can download Python [here](https://www.python.org/downloads/).

## Set Up the Environment

1. **Clone the Repository**: Open a terminal and run the following commands:
    ```bash
    git clone https://github.com/nisakhantalib/fyp.git
    cd fyp
    ```

2. **Create and Activate a Virtual Environment**:
    - For Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
        If you encounter any warnings, you can run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` and then try activating the environment again.

    - For macOS and Linux:
        ```bash
        python -m venv venv
        source venv/bin/activate
        ```

3. **Install Dependencies**: Install the required dependencies by running:
    ```bash
    pip install -r requirements.txt
    ```

## Prepare the Data
- The dataset `50human100eachgpt4withnumber.xlsx` is included in the repository. For more information on data extraction and GPT-4 text generation, run the following scripts:
    ```bash
    python autoextraction.py
    python promptingGPT4.py
    ```

## Models Download
- The trained models are hosted on Google Drive due to their large size. Please download them from the following [link](https://drive.google.com/drive/folders/1kG_oZCJgX7tpPZRCKc6-sQmQ9rc9RJeO?usp=sharing).
- After downloading, unzip the files and place them in the `models/` directory in the project folder.

## Training and Saving the Models
- If you prefer to train and save the models manually, execute the `submission.ipynb` script. It is recommended to run this in a Jupyter environment, such as Google Colab or Jupyter Notebook/Lab.

## Running the Web Application
- Start the web application by running:
    ```bash
    python app.py
    ```
- Open a web browser and navigate to `http://localhost:5000` to view the application.
