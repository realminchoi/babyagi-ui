<h1 align="center">
 babyagi-ui

</h1>

# Introduction
This Python script is non-poetry version of the [babyagi-streamlit](https://github.com/dory111111/babyagi-streamlit) for those not familiar with poetry setup/run.  This code base is inspired by BabyAGI, see Acknowledgements section below.

# Demo
[streamlit-babyagi-2023-04-09-20-04-52.webm](https://user-images.githubusercontent.com/67872688/230803873-b744c9e2-d516-4e5d-9ef2-67f934b9b35c.webm)

# Installation
Install the required packages:
````
pip install -r requirements.txt
````
## Usage

Run streamlit.
````
python -m streamlit run babyagi.py 
````

You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501

To stop the Streamlit server, press ctrl-C.

If you get the following popup error due to server being stopped accidentally, manually, or in some way

![Connection Error](img/conn_error.png)

Just re-run streamlit with the following command.
````
python -m streamlit run babyagi.py 
````

# Using Poetry for Installation and Usage
If you want to use Poetry to install and run you can follow this instruction.

Install the required packages:
````
poetry install
````

## Usage

Run streamlit.
````
poetry run streamlit run babyagi.py 
````

# Acknowledgments

I would like to express my gratitude to the developers whose code I referenced in creating this repo.

Special thanks go to 

@yoheinakajima (https://github.com/yoheinakajima/babyagi)

@hinthornw (https://github.com/hwchase17/langchain/pull/2559)

@dory111111 (https://github.com/dory111111/)

---
Roboto Logo Icon by Icons Mind(https://iconscout.com/contributors/icons-mind) on IconScout(https://iconscout.com)
