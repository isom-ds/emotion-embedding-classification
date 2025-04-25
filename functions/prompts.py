# Define the prompt generation functions
def generate_prompt(data_point):
    return f"""
Classify the text into {', '.join(l_emotions)}
Return the answer as the corresponding emotion label.
text: {data_point["text"]}
label: {data_point["emotions"]}""".strip()

def generate_test_prompt(data_point):
    return f"""
Classify the text into {', '.join(l_emotions)}
Return the answer as the corresponding emotion label.
text: {data_point["text"]}
label: """.strip()