# 🎯 Mentor Recommendation System
---
This repository contains a recommendation system built to recommend mentors based on course preferences. The project consists of three main Python scripts, where `mentor_recommendation_app.py` serves as the main script for deployment via Streamlit.

---
## Objective
This project aims to design a simple AI/ML-based solution to recommend mentors (CLAT toppers) to law aspirants based on their user profiles. The system will consider various factors such as the aspirant's preferred subjects, target colleges, current preparation level, and learning style. The recommendation system is built using basic classification, clustering, or recommendation techniques.

### Problem Statement
The goal of this project is to design a mentor recommendation system that can suggest the top 3 mentors to law aspirants, particularly for those preparing for CLAT and similar entrance exams. This is achieved by processing features like:
- Preferred subjects
- Target colleges
- Current preparation level
- Learning style

The system uses techniques such as KNN (K-Nearest Neighbors), content-based filtering, or cosine similarity to provide personalized mentor recommendations.

## Approach
The mentor recommendation system processes the following features from the aspirants' profiles:
- **Preferred subjects**: Subjects the aspirant is interested in for their law exams.
- **Target colleges**: Colleges the aspirant is aiming for.
- **Current preparation level**: The aspirant's preparation level (beginner, intermediate, advanced).
- **Learning style**: Preferences regarding learning methodologies (e.g., one-on-one mentoring, group study, video tutorials).

Using these features, the system employs **content-based filtering** and **cosine similarity** techniques to recommend mentors who are best suited to the aspirant's profile. The recommendations are personalized based on how closely the mentor's profile matches the aspirant's preferences.

### Expected Output
The system will output the top 3 mentor recommendations based on the user's profile input. For example:

- **Mentor 1**: John Doe (Specializes in Constitutional Law, Target College: National Law University)
- **Mentor 2**: Jane Smith (Specializes in Criminal Law, Target College: NLSIU Bangalore)
- **Mentor 3**: Rahul Verma (Specializes in Family Law, Target College: NLU Delhi)

The system provides details about each mentor, including their area of expertise, years of experience, and mentoring style.

## Project Structure
```bash
mentor-recommendation-system/ 
│ └── mentor_recommendation_app.py # (Streamlit app containing all the app code) 
├── data/
│   └── mock_mentors.csv # (Course data or any relevant datasets used)
│   └── mock_profiles.csv # (Course data or any relevant datasets used)
├── requirements.txt # (List of dependencies for the project) 
└── README.md # (This file)
```

## Description

This project recommends mentors based on course preferences, leveraging a recommendation model built using course data. The recommendation system includes:
- Scraping relevant course data.
- Generating embeddings for course information.
- Using these embeddings to recommend mentors based on user preferences.

## Setup

Follow the steps below to set up the project locally or deploy it.

### 1. Clone the Repository

```bash
git clone https://github.com/yogesh43221/mentor-recommendation-system.git
cd mentor-recommendation-system
```

### 2. Install Dependencies
It is recommended to create a virtual environment and install the necessary dependencies.
```bash
# Create virtual environment (optional but recommended)
python -m venv venv
# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On MacOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
### 3. Run the Scripts Locally
  * You can run the three Python scripts directly in your local VS Code terminal, which are:
  * scrape_courses.py: Scrapes course data from the specified sources (if applicable).
  * generate_embeddings.py: Generates embeddings for courses to facilitate recommendations.
  * recommendation_model.py: The main logic for generating mentor recommendations based on user input.

To run these scripts, simply use:
```bash
python scripts/scrape_courses.py
python scripts/generate_embeddings.py
python scripts/recommendation_model.py
```
You can modify or enhance these scripts to fit your specific needs or sources.

### 4. Main File for Deployment
  * The main script for deploying the recommendation system using Streamlit is ```mentor_recommendation_app.py```.

To deploy the app locally:
```bash
streamlit run app/mentor_recommendation_app.py
```
This will launch a Streamlit app in your browser where you can interact with the system, input your preferences, and receive mentor recommendations.

### 5. Requirements
All dependencies for the project are listed in the ```requirements.txt``` file. If you're setting up the project, ensure you have all required libraries installed.

Example ```requirements.txt```:
```bash
streamlit
pandas
scikit-learn
numpy
```
### How It Works
The user inputs their profile information, including:
- Preferred subjects
- Target colleges
- Current preparation level
- Learning style

The `mentor_recommendation_app.py` Streamlit app takes these inputs and processes them using the models and embeddings generated by the other scripts. Based on the input, it provides a list of the top 3 mentor recommendations tailored to the user's preferences.

To run the app locally:
```bash
streamlit run app/mentor_recommendation_app.py
```

## Contribution
Feel free to fork the repository and submit pull requests for any enhancements or bug fixes. You can also open issues for any problems or feature requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

### Key Points:
- **`mentor_recommendation_app.py`** is the main file for deployment with Streamlit.
- The other scripts (`scrape_courses.py`, `generate_embeddings.py`, `recommendation_model.py`) are complementary and can be run in your local VS Code terminal to handle different stages of the recommendation pipeline.
- The `requirements.txt` contains all the dependencies required to run the project.

This structure and instructions will make it easy for someone else (or you in the future) to run, deploy, and contribute to the project. Let me know if you need any further changes!
