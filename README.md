# 🎯 AI Interview Coach (simple version)

Practise a job interview by chatting with an AI **senior from your field**.
Enter a role or upload your CV, say hi, answer the questions, and get feedback after each answer.
Click **Finish interview** for a final report you can download.

## How to run
```bash
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env           # Mac/Linux: cp .env.example .env
# open .env and paste your free key from https://console.groq.com/keys
streamlit run app.py
```

## How it works (multi-step)
The whole app is **one chat window**. The message box has a 📎 button to upload a CV.
1. **Read input:** type a role, or upload one CV (PDF). With only a CV, the AI picks the best-fitting role.
2. **Generate questions** tailored to the role/CV.
3. **Chat:** the senior introduces themselves with tips, then asks question 1.
4. For every message:
   - an answer → score, what was good, what is missing, a better answer
   - a question back (ends with `?`) → explanation, no score
   - `skip` → next question
5. **Finish** → final report (score, strengths, improvements, 1-week plan) + download.

## Prompt techniques
| Technique | Where in `app.py` | Why |
|---|---|---|
| **Persona** | `persona()` | The AI acts as a senior in your field, so tone and expectations fit the job |
| **Few-shot** | `QUESTION_PROMPT` | Example questions show the style and format we want |
| **Chain-of-Thought** | `FEEDBACK_PROMPT` | The AI reasons step by step before giving a score, so feedback is fairer |

## Parameters (sidebar sliders)
| Parameter | Controls | Low | High |
|---|---|---|---|
| temperature | creativity | safe, similar replies | creative, varied |
| max_tokens | reply length | short (may be cut off) | long |
| top_p | probability threshold for word choice | only the most likely words | more varied wording |

## Error handling
Every check is simple: catch the problem and print a ⚠️ message in the chat.

| Situation | Message |
|---|---|
| 2 or more CVs uploaded | "Please upload only 1 CV." |
| 2 or more roles typed ("Data Analyst and Software Engineer") | "Which role do you want to focus on? Please choose one." |
| Scanned / damaged PDF | "I can't read this PDF..." |
| Only "hi" typed | Asks for a job role |
| Very long text instead of a job title | "Please type only the job title" |
| CV uploaded during the interview | "Click Start over to use a new CV" |
| Empty answer | "Your answer is empty" |
| Finish before answering anything | "Answer at least one question first" |
| Missing / wrong API key | Points to the .env file |
| Rate limit, timeout, no internet | "Wait one minute / send your message again" |
| Empty AI reply | "Try raising Max tokens" |
