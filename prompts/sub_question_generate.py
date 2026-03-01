"""
Sub-Question Generation Prompt — Vgent-style Structured Reasoning

Breaks down a multiple-choice question into yes/no sub-questions
that can be verified independently against each video segment.

Based on Vgent's SQL_PROMPT.
"""

PROMPT = """Given a question about a long video and potential candidates:
Question: {question}

Candidates: {candidates}

Given a multiple-choice question about a video, break it down into several sub-questions that analyze the key elements required to answer it step by step.

Form yes/no or counting questions to verify the presence of the subject or event in the video (e.g., "Does the video show [subject/event]?").
Ensure the sub-questions cover all necessary aspects to reach the correct answer.

Example 1:
Question: Which of the following statements is not correct?
Candidates:
A. The Titanic finally sank because 5 adjacent compartments were breached.
B. Despite the lack of lifeboats, the Titanic met all the requirement.
C. People on the Titanic were not rescued in time because its operator was sleeping.
D. The Titanic was equipped with 20 lifeboats.
Your output:
{{
    "Q1": "Does the video show the Titanic finally sank because 5 adjacent compartments were breached?",
    "Q2": "Does the video show the Titanic met all the requirement despite the lack of lifeboats?",
    "Q3": "Does the video show people on the Titanic were not rescued in time because its operator was sleeping?",
    "Q4": "Does the video show the Titanic was equipped with 20 lifeboats?"
}}

Example 2:
Question: How many ships are shown in the map while the sinking ship sending out message?
Candidates: A. 3. B. 8. C. 11. D. 9.
Your output:
{{
    "Q1": "Is there a map showing a sinking ship sending out a message?"
}}

Example 3:
Question: What is the score at the end of the half?
Candidates: A. 38 - 31. B. 38 - 34. C. 67 - 61. D. 67 - 60.
Your output:
{{
    "Q1": "Is the video showing the game at the end of the half?"
}}

Example 4:
Question: Which athlete in the video was the first to touch off the crossbar?
Candidates: A. Athlete from Russia. B. Athlete from Qatar. C. Athlete from Canada. D. Athlete from Ukraine.
Your output:
{{
    "Q1": "Is the athlete from Russia touch off the crossbar?",
    "Q2": "Is the athlete from Qatar touch off the crossbar?",
    "Q3": "Is the athlete from Canada touch off the crossbar?",
    "Q4": "Is the athlete from Ukraine touch off the crossbar?"
}}

Example 5:
Question: According to the video, what is the chronological order in which the following actions occur?
(a) Weaving in the ends. (b) Crocheting a single crochet. (c) Finishing the handcraft. (d) Making a slip knot. (e) Crocheting a chain.
Your output:
{{
    "Q1": "Is there a scene showing weaving in the ends?",
    "Q2": "Is there a scene showing crocheting a single crochet?",
    "Q3": "Is there a scene showing finishing the handcraft?",
    "Q4": "Is there a scene showing making a slip knot?",
    "Q5": "Is there a scene showing crocheting a chain?"
}}

Output ONLY valid JSON with sub-questions. Do not include any other text."""
