import streamlit as st

from langchain_groq import ChatGroq

from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.callbacks import StreamlitCallbackHandler



st.set_page_config(
    page_title="Math & Research Assistant",
    page_icon="🤖"
)

st.title("🤖 Math & Research Assistant")

groq_api_key = st.sidebar.text_input(
    "Enter GROQ API Key",
    type="password"
)

if not groq_api_key:
    st.info("Please enter your GROQ API key.")
    st.stop()



llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)



wiki = WikipediaAPIWrapper()

wikipedia_tool = Tool(
    name="Wikipedia",
    func=wiki.run,
    description="""
    Useful for factual information, history,
    people, places, companies and general knowledge.
    """
)



def calculator_tool(expression: str):
    try:
        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )
        return str(result)
    except Exception as e:
        return f"Calculation Error: {e}"

calculator = Tool(
    name="Calculator",
    func=calculator_tool,
    description="""
    Useful for mathematical calculations.

    Input examples:
    25*7
    500*0.25
    (45+10)/5
    2**8

    Only send valid mathematical expressions.
    """
)


template = """
You are an expert reasoning assistant.

Solve the problem carefully.

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=template
)

reasoning_chain = LLMChain(
    llm=llm,
    prompt=prompt
)

reasoning_tool = Tool(
    name="Reasoning",
    func=reasoning_chain.run,
    description="""
    Useful for logic questions,
    explanations and step-by-step reasoning.
    """
)


assistant_agent = initialize_agent(
    tools=[
        wikipedia_tool,
        calculator,
        reasoning_tool
    ],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! Ask me math, reasoning, or research questions."
        }
    ]

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(
        message["content"]
    )



question = st.text_area(
    "Enter your question:",
    placeholder="What is 25% of 500?"
)


if st.button("Find My Answer"):

    if question.strip():

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        st.chat_message("user").write(question)

        with st.spinner("Thinking..."):

            st_cb = StreamlitCallbackHandler(
                st.container(),
                expand_new_thoughts=False
            )

            response = assistant_agent.run(
                question,
                callbacks=[st_cb]
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )

        st.chat_message("assistant").write(response)

    else:
        st.warning("Please enter a question.")

