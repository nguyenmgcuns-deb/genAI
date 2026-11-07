# AI Interview Coach
# A chatbot that helps you prepare for a job interview.
# Run with: streamlit run app.py

import os

import streamlit as st
from dotenv import load_dotenv
from groq import Groq, RateLimitError, APITimeoutError, APIConnectionError, AuthenticationError
from pypdf import PdfReader

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b"
NUM_QUESTIONS = 5

# Each task gets its own settings.
# Low temperature = precise and repeatable, high temperature = more creative.
# max_tokens limits the length of the reply, top_p limits which words the model can choose.
SETTINGS = {
    "extract":   {"temperature": 0.0, "max_tokens": 1000, "top_p": 1.0},   # read role from text
    "analyze":   {"temperature": 0.2, "max_tokens": 2000, "top_p": 0.9},   # CV analysis
    "questions": {"temperature": 0.8, "max_tokens": 2000, "top_p": 0.95},  # varied questions
    "advice":    {"temperature": 0.6, "max_tokens": 2000, "top_p": 0.9},
    "feedback":  {"temperature": 0.3, "max_tokens": 2000, "top_p": 0.9},   # fair, consistent
    "summary":   {"temperature": 0.4, "max_tokens": 2500, "top_p": 0.9},
}

st.set_page_config(page_title="AI Interview Coach", page_icon="🎯")

if not API_KEY:
    st.error("No API key found. Add GROQ_API_KEY to your .env file.")
    st.stop()

client = Groq(api_key=API_KEY, timeout=30)


# ------------------------------------------------------------
# Prompts
# ------------------------------------------------------------

# Persona: the model acts as a senior from the user's field
def persona(role, company=""):
    where = f" at a company like {company}" if company else ""
    return (f"You are Alex, a friendly senior {role}{where} with 15 years of experience. "
            f"You help candidates prepare for {role} interviews. Speak simply and clearly.")


# One-shot: one example shows how to read role, company and question from a sentence
ROLE_PROMPT = """Find the job role, the company and any question in the user's message.

Example:
Message: "hi, I'm a student and I want to apply as a data analyst intern at BMW Debrecen. How should I research the company?"
ROLE: Data Analyst Intern
COMPANY: BMW Group Plant Debrecen
QUESTION: How should I research the company?

If the user wants several different jobs, put them all on the ROLE line separated by ";".
Write "none" for anything that is missing. Reply with the 3 lines only.

Message: "{text}" """

# Document analysis: turn the CV text into a structured review
CV_PROMPT = """Analyse this CV for a {role} application.
Only use what is written in the CV, do not invent anything.

Reply in Markdown with these sections:
**Summary:** 2 sentences about the candidate
**Key skills:** up to 6 skills
**Strengths for this role:** 3 bullets
**Gaps to prepare for:** 3 bullets (things the role needs that the CV does not show)
**Keywords to use in the interview:** 5-8 keywords

CV:
{cv}"""

# Few-shot: two examples show the style of good advice
ADVICE_PROMPT = """The candidate is preparing for a {role} interview{company}.
Answer like a mentor: practical steps, short bullets, max 150 words.
If they ask about a specific company, only share facts you are sure about and tell them
where to check (company website, LinkedIn, recent news). Never invent facts.

Example 1
Question: How do I introduce myself?
Answer: Use 3 parts, about 1 minute:
- Present: what you study or do now
- Past: 1-2 experiences that fit this job
- Future: why this role and this company

Example 2
Question: What should I research about the company?
Answer:
- Products and customers: what they make and for whom
- Recent news: new projects or plants (check their newsroom)
- Values: read the careers page and connect one value to your own story

Question: {question}
Answer:"""

# Few-shot: examples of good interview questions
QUESTION_PROMPT = """Write {n} interview questions for a {role}{company}.
Base them on this CV if there is one: {cv}

Examples of good questions (for a different job, Backend Developer):
- Your CV mentions a REST API. How did you handle user login?
- Tell me about a time a teammate disagreed with you. What did you do?

Mix technical and behavioural questions, from easy to hard.
Write only the questions, one per line, without numbers."""

# Chain-of-thought: the model reasons step by step before giving feedback
FEEDBACK_PROMPT = """Question: {question}
Candidate's answer: {answer}

Think step by step:
1. What is this question testing?
2. What would a strong answer include?
3. What did the candidate do well, and what is missing?

Then reply in Markdown (no score or points):
**Good:** 2 bullets
**Missing:** 2 bullets
**How to improve:** 2 short tips
**Better answer:** a short example answer, max 80 words"""

HINT_PROMPT = """Interview question: {question}
The candidate is stuck. Give a hint but do not write the answer:
- one sentence on what the question is testing
- 2-3 keywords a good answer should mention
- a simple structure to follow (for example situation, action, result)
Max 70 words."""

SUMMARY_PROMPT = """Here is a conversation between a candidate and an interview coach.
Write a short summary in Markdown with these sections:
## What we covered (2-3 bullets)
## Most important advice (3 bullets)
## Keywords to keep in mind (6-8 keywords, each with a few words of explanation)
## Next steps (3 short steps)
Keep it under 250 words.

Conversation:
{chat}"""


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def say(text):
    st.session_state.messages.append(("assistant", text))


def ask_ai(system, prompt, task):
    """Send one request to the API. Errors are caught and shown in the chat."""
    settings = SETTINGS[task]
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            temperature=settings["temperature"],
            max_completion_tokens=settings["max_tokens"],
            top_p=settings["top_p"],
        )
        reply = response.choices[0].message.content
        if not reply:
            say("⚠️ The AI sent back an empty answer. Please try again.")
        return reply
    except AuthenticationError:
        say("⚠️ The API key is not valid. Please check the .env file.")
    except RateLimitError:
        say("⚠️ Too many requests at once. Wait a minute and send your message again.")
    except (APITimeoutError, APIConnectionError):
        say("⚠️ Could not connect to the AI. Check your internet and try again.")
    except Exception as error:
        say(f"⚠️ Something went wrong: {error}")
    return None


def read_cv(file):
    """Return the text of the CV, or None if there are no words in it."""
    if file.name.lower().endswith((".png", ".jpg", ".jpeg")):
        return None
    try:
        pages = PdfReader(file).pages
        text = " ".join(page.extract_text() or "" for page in pages)
    except Exception:
        return None
    if len(text.split()) < 20:      # a scanned page gives almost no text
        return None
    return text[:8000]


def find_role(text):
    """Returns (roles, company, question) from the user's sentence."""
    reply = ask_ai("You extract information exactly as asked.", ROLE_PROMPT.format(text=text), "extract")
    if reply is None:
        return None

    found = {"ROLE": "", "COMPANY": "", "QUESTION": ""}
    for line in reply.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip(" *").upper()
        value = value.strip(" *\"")
        if key in found and value.lower() != "none":
            found[key] = value

    roles = [r.strip() for r in found["ROLE"].split(";") if r.strip()]
    return roles, found["COMPANY"], found["QUESTION"]


def company_text():
    if st.session_state.company:
        return f" at {st.session_state.company}"
    return ""


def current_question():
    qs = st.session_state.questions
    i = st.session_state.current
    if st.session_state.mode == "practice" and i < len(qs):
        return qs[i]
    return None


def next_question_text():
    qs = st.session_state.questions
    i = st.session_state.current
    if i < len(qs):
        return f"\n\n**Question {i + 1}/{len(qs)}:** {qs[i]}"
    return "\n\nThat was the last question! You can click **Summarize** to get your summary."


# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.role = ""
    st.session_state.company = ""
    st.session_state.cv = ""
    st.session_state.mode = ""          # "advice" or "practice"
    st.session_state.questions = []
    st.session_state.current = 0
    st.session_state.summary = ""
    say("Hi! I'm your interview coach. 👋\n\n"
        "Tell me which job you're preparing for, in your own words "
        "(for example *\"I'm an intern applying for a data analyst job at BMW\"*), "
        "or upload your CV with the 📎 button.")


# ------------------------------------------------------------
# Actions
# ------------------------------------------------------------

def set_role(role, company=""):
    st.session_state.role = role
    st.session_state.company = company
    where = f" at **{company}**" if company else ""
    say(f"Great, let's prepare you for **{role}**{where}.\n\n"
        "Choose what you want to do with the buttons below:\n"
        "- **Advice**: ask me anything, like how to research the company\n"
        "- **Practice questions**: I ask interview questions and give you feedback")


def analyze_cv(cv_text):
    """Document analysis pipeline: read the PDF, find the role if needed, then review the CV."""
    st.session_state.cv = cv_text
    role = st.session_state.role
    if not role:
        role = ask_ai("You read CVs.", "Which one job title fits this CV best? "
                      "Reply with the job title only.\n\n" + cv_text, "extract")
        if not role:
            return
        role = role.strip(" .*\"")

    review = ask_ai(persona(role), CV_PROMPT.format(role=role, cv=cv_text), "analyze")
    if review:
        say("📄 **CV analysis**\n\n" + review)
        set_role(role, st.session_state.company)


def handle_first_message(text, files):
    if len(files) > 1:
        say(f"⚠️ You uploaded {len(files)} files. Please upload only 1 CV.")
        return

    if len(files) == 1:
        cv_text = read_cv(files[0])
        if cv_text is None:
            say("⚠️ I couldn't find any words in this file. It looks like a picture or a scanned page.\n\n"
                "Please upload your CV again as a PDF that contains text.")
            return
        if text:   # they also wrote the job, read it first
            result = find_role(text)
            if result and len(result[0]) == 1:
                st.session_state.role = result[0][0]
                st.session_state.company = result[1]
        analyze_cv(cv_text)
        return

    if not text:
        say("⚠️ Please tell me which job you're preparing for, or upload your CV.")
        return

    if text.lower().strip("!. ") in ["hi", "hello", "hey", "good morning"]:
        say("Hello! 😊 Which job are you preparing for? You can also upload your CV.")
        return

    if len(text) > 1500:
        say("⚠️ That message is very long. Please describe the job in a few sentences, or upload your CV.")
        return

    result = find_role(text)
    if result is None:
        return
    roles, company, question = result

    if not roles:
        say("🤔 I couldn't find a job in your message. Which job are you preparing for?")
        return

    if len(roles) > 1:
        options = "\n".join(f"- {role}" for role in roles)
        say(f"⚠️ You mentioned {len(roles)} roles:\n{options}\n\n"
            "Which one do you want to focus on? Please choose one.")
        return

    set_role(roles[0], company)
    if question:
        st.session_state.mode = "advice"
        give_advice(question)


def give_advice(question):
    prompt = ADVICE_PROMPT.format(role=st.session_state.role, company=company_text(), question=question)
    reply = ask_ai(persona(st.session_state.role, st.session_state.company), prompt, "advice")
    if reply:
        say(reply)


def start_practice():
    qs = st.session_state.questions
    if not qs or st.session_state.current >= len(qs):
        prompt = QUESTION_PROMPT.format(n=NUM_QUESTIONS, role=st.session_state.role,
                                        company=company_text(), cv=st.session_state.cv or "no CV")
        reply = ask_ai(persona(st.session_state.role, st.session_state.company), prompt, "questions")
        if reply is None:
            return

        questions = []
        for line in reply.splitlines():
            line = line.strip("-•* 0123456789.").strip()
            if len(line) > 10:
                questions.append(line)
        if not questions:
            say("⚠️ I couldn't create the questions. Please click Practice questions again.")
            return

        st.session_state.questions = questions[:NUM_QUESTIONS]
        st.session_state.current = 0

    st.session_state.mode = "practice"
    say("Let's practise! Answer in your own words. If something is unclear, just ask me."
        + next_question_text())


def handle_answer(text):
    question = current_question()

    if text.lower() == "skip":
        skip_question()
        return

    if text.endswith("?"):   # the user asks about the question instead of answering
        prompt = (f"Interview question: {question}\nThe candidate asks: {text}\n"
                  "Answer briefly without giving away the answer.")
        reply = ask_ai(persona(st.session_state.role), prompt, "advice")
        if reply:
            say(reply)
        return

    prompt = FEEDBACK_PROMPT.format(question=question, answer=text)
    reply = ask_ai(persona(st.session_state.role, st.session_state.company), prompt, "feedback")
    if reply:
        st.session_state.current += 1
        say(reply + next_question_text())


def skip_question():
    st.session_state.current += 1
    say("No problem, let's move on." + next_question_text())


def give_hint():
    reply = ask_ai(persona(st.session_state.role), HINT_PROMPT.format(question=current_question()), "advice")
    if reply:
        say("💡 **Hint:**\n\n" + reply)


def summarize():
    chat = "\n\n".join(f"{who.upper()}: {text}" for who, text in st.session_state.messages)
    reply = ask_ai(persona(st.session_state.role), SUMMARY_PROMPT.format(chat=chat[-12000:]), "summary")
    if reply:
        st.session_state.summary = reply
        say("📝 **Summary**\n\n" + reply)


# ------------------------------------------------------------
# Page layout
# ------------------------------------------------------------

# Start over is pinned to the top-right corner, so it stays visible while scrolling
st.markdown("""
<style>
.st-key-start_over {
    position: fixed;
    top: 6rem;
    right: max(1rem, calc(50vw - 22rem));   /* right edge of the centered chat area */
    width: auto;
    z-index: 1000;
}
</style>
""", unsafe_allow_html=True)

with st.container(key="start_over"):
    if st.button("🔄 Start over"):
        st.session_state.clear()
        st.rerun()

st.title("AI Interview Coach")

for who, text in st.session_state.messages:
    st.chat_message(who).markdown(text)

message = st.chat_input("Write here, or upload your CV with 📎",
                        accept_file="multiple", file_type=["pdf", "png", "jpg", "jpeg"])

if message:
    text = (message.text or "").strip()
    files = message.files or []
    shown = text
    if files:
        shown += "\n\n📎 " + ", ".join(f.name for f in files)
    st.session_state.messages.append(("user", shown or "(empty message)"))

    with st.spinner("Thinking..."):
        if not st.session_state.role:
            handle_first_message(text, files)
        elif files:
            say("⚠️ I already have your details. Click **Start over** to use a different CV.")
        elif not text:
            say("⚠️ Your message is empty.")
        elif current_question():
            handle_answer(text)
        else:
            st.session_state.mode = "advice"
            give_advice(text)
    st.rerun()

if st.session_state.role:
    on_question = current_question() is not None

    col1, col2 = st.columns(2)
    if col1.button("🧭 Advice", use_container_width=True,
                   type="primary" if st.session_state.mode == "advice" else "secondary"):
        st.session_state.mode = "advice"
        say("Advice mode. Ask me anything about preparing for this job, "
            "for example *\"How should I research the company?\"*")
        st.rerun()
    if col2.button("🎤 Practice questions", use_container_width=True,
                   type="primary" if st.session_state.mode == "practice" else "secondary"):
        with st.spinner("Preparing questions..."):
            start_practice()
        st.rerun()

    col3, col4, col5 = st.columns(3)
    if col3.button("💡 Hint", use_container_width=True, disabled=not on_question):
        st.session_state.messages.append(("user", "Can I have a hint?"))
        with st.spinner("Thinking..."):
            give_hint()
        st.rerun()
    if col4.button("⏭️ Skip", use_container_width=True, disabled=not on_question):
        st.session_state.messages.append(("user", "Skip this question"))
        skip_question()
        st.rerun()
    if col5.button("📝 Summarize", use_container_width=True):
        user_messages = [m for m in st.session_state.messages if m[0] == "user"]
        if len(user_messages) < 2:
            st.warning("Ask for some advice or answer a question first, then I can summarize.")
        else:
            with st.spinner("Writing the summary..."):
                summarize()
            st.rerun()

if st.session_state.summary:
    st.download_button("⬇️ Download summary", st.session_state.summary, "summary.md")
