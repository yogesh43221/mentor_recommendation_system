import streamlit as st
import pandas as pd

# Load mentor data
mentors_df = pd.read_csv("data/mock_mentors.csv")

# User Input Section
st.title("🎯 Mentor Recommendation System")
st.subheader("Provide your preferences:")

# Dynamic dropdowns from mentor dataset
all_subjects = sorted({subj for s in mentors_df['top_subjects'] for subj in s.split('|')})
learning_styles = mentors_df['learning_style'].unique().tolist()
colleges = mentors_df['mentor_college'].unique().tolist()
prep_levels = ['Beginner', 'Intermediate', 'Advanced']
study_times = ['1 hour', '1.5 hours', '2 hours', '3 hours', '4 hours', '5 hours']
goals = ['Top 10 Law Colleges', 'Top 5 Law Colleges', 'Top 3 Law Colleges']
experience_levels = ['1-2 years', '2-3 years', '3+ years']

# Input fields
preferred_subjects = st.multiselect("Preferred Subjects", all_subjects, default=["English", "Logical Reasoning"])
learning_style = st.selectbox("Preferred Learning Style", learning_styles)
target_college = st.selectbox("Target College", colleges)
prep_level = st.selectbox("Preparation Level", prep_levels)
study_time = st.selectbox("Study Time Per Day", study_times)
goal = st.selectbox("Goal", goals)
preferred_exp = st.selectbox("Preferred Mentor Experience", experience_levels)

# Filter options
only_available = st.checkbox("Only show available mentors", value=True)
min_experience = st.slider("Minimum Years of Experience", 0, 5, 1)

# Button to trigger recommendation
if st.button("Get Recommendations"):
    def match_score(row):
        score = 0
        reasons = []

        mentor_subjects = row['top_subjects'].split('|')

        if any(subj in mentor_subjects for subj in preferred_subjects):
            score += 1
            reasons.append("✅ Subjects")
        else:
            reasons.append("❌ Subjects")

        if row['learning_style'] == learning_style:
            score += 1
            reasons.append("✅ Learning Style")
        else:
            reasons.append("❌ Learning Style")

        if row['mentor_college'] == target_college:
            score += 1
            reasons.append("✅ College Match")
        else:
            reasons.append("❌ College Match")

        if row['years_of_experience'] >= min_experience:
            score += 1
            reasons.append("✅ Experience")
        else:
            reasons.append("❌ Experience")

        return score, ", ".join(reasons)

    # Filter mentor list if needed
    filtered_df = mentors_df.copy()
    if only_available:
        filtered_df = filtered_df[filtered_df['mentor_availability'] == "Available"]

    # Calculate score and reason
    filtered_df[['score', 'match_reason']] = filtered_df.apply(lambda row: pd.Series(match_score(row)), axis=1)

    # Sort by score and rating
    sorted_df = filtered_df.sort_values(by=['score', 'mentee_feedback_rating'], ascending=False).head(3)

    # Display Recommendations
    st.subheader("Top 3 Recommended Mentors")

    cols = st.columns(3)

    for idx, (i, row) in enumerate(sorted_df.iterrows()):
        with cols[idx]:
            st.markdown(f"### 👤 Mentor {row['mentor_id']}")
            st.markdown(f"**Top Subjects:** {row['top_subjects']}")
            st.markdown(f"**Learning Style:** {row['learning_style']}")
            st.markdown(f"**Mentoring Style:** {row['mentoring_style']}")
            st.markdown(f"**College:** {row['mentor_college']}")
            st.markdown(f"**Years of Experience:** {row['years_of_experience']}")
            st.markdown(f"**Availability:** {row['mentor_availability']}")
            st.markdown(f"**Feedback Rating:** ⭐ {row['mentee_feedback_rating']}")
            st.markdown(f"**Match Score:** {row['score']} / 4")
            st.markdown(f"**Match Reasons:** {row['match_reason']}")

