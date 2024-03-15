To run the web application locally on your machine, follow these steps:

Prerequisites
Ensure Python 3 is installed on your machine.

Download or clone this repository to your local machine.

Set Up the Environment
Open a terminal (Command Prompt on Windows, Terminal on macOS and Linux).

Navigate to the project's root directory using the cd command. Replace <project-directory> with the path to your project directory:

bash
Copy code
cd <project-directory>

Create a virtual environment (this keeps your dependencies organized and does not conflict with other projects):

bash
Copy code
python -m venv venv
Activate the virtual environment:

On Windows:
bash
Copy code
.\venv\Scripts\activate
On macOS and Linux:
bash
Copy code
source venv/bin/activate
Install the required dependencies using the following command:

bash
Copy code
pip install -r requirements.txt

Prepare the Data
The dataset used for training and testing model , 50human100eachgpt4withnumber.xlsx is already available in the repository. 
The input dataset used is the reuters5050 dataset, downloaded into drive. for more info on extracting text and prompting gpt4 to produce text,run the following scripts:
bash
Copy code
python autoextraction.py
python promptingGPT4.py

Saved Model
Visit the following google drive link: https://drive.google.com/drive/folders/1kG_oZCJgX7tpPZRCKc6-sQmQ9rc9RJeO?usp=drive_link
Download the required model files to your local machine. Note: you may need to download each file individually or download the entire folder as a zip file (right-click on the folder and select "Download").

Once downloaded, move the model files to the appropriate directory in your project's root folder.Create a subfolder called 'models' and move all the downloaded models into the subfolder. 

Training and Saving the Models
To manually train and save the models, run the following script. Note that this script refers to a Jupyter notebook. It is recommended to run this in a Jupyter environment like Google Colab or Jupyter Lab/Notebook:
bash
Copy code
jupyter notebook submission.ipynb

Running the Web Application
To start the web server and run the web application, execute:

bash
Copy code
python app.py
Open a web browser and go to http://localhost:5000 to view the application.