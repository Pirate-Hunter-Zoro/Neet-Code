# About
This is a repository containing my implementation of various problems and data structure implementations from NeetCode in Python. My instructions apply only to Mac and Linux users - Windows users are unfortunately on their own in getting this code to run on their machines!
I am assuming you have Homebrew installed

## Instruction
1. Navigate to a folder on your computer where you would like to clone this project via the command line and run the following command:<br>
   ```git clone https://github.com/Pirate-Hunter-Zoro/Neet-Code.git```<br>
This will create a folder called "Neet-Code" in your current folder. 
2. Navigate into this folder.<br>
   ```cd Neet-Code```
3. Create a python virtual environment named "neet-code-env" (or whatever you want to call it) via the following commands (We'll need an older version of python so that we can use the pytorch library):<br>
   ```brew install python@3.10```<br>
   ```python3.10 -m venv venv```
4. Activate the virtual environment:<br>
   ```source venv/bin/activate```
5. Install the necessary Python libraries:<br>
   ```pip install -r requirements.txt```
6. At this point, you should be able to allow VSCode to configure your Python tests and you can run them should you choose to!

## Useful Note
To write all of the requirements in the Python virtual library into "requirements.txt", run the following command:<br>
```pip freeze > requirements.txt```
