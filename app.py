import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

st.set_page_config(page_title="Math Solver Assistant", page_icon="🦜")
st.title("Text to Math Problem Solver")

groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not groq_api_key:
    st.info("Please add your Groq API key to continue")
    st.stop()

llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

## Tools
wiki_wrapper = WikipediaAPIWrapper()
wiki_tools = Tool(
    name="Wikipedia",
    func=wiki_wrapper.run,
    description="a Tool for searching the internet to find the various information on the topics mentioned"
)

## Math Tool
math_chain = LLMMathChain.from_llm(llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="a tool for answering Math related question. Only input mathematical expression need to be provided"
)

prompt = """
Your a agent tasked for solving users mathematical question. Logically arrive at the solution and provide a detailed
explanation and display it point wisefor the question below
Question:{question}
Answer:
"""

promp_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

## Combine all Tool

chain=LLMChain(llm=llm, prompt=promp_template)

reasoning_tool = Tool(
    name="Reasoning",
    func=chain.run,
    description="a Tool for answering logic-based and reasoning question."
)

## Initialize Agent
assistant_agent = initialize_agent(
    tools=[wiki_tools, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handling_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state['messages']= [
        {"role":"assistant", "content":"Hi, I'm a MATH chatbot"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

## start interaction
question=st.text_area("Enter your question")

if st.button("find my answer"):
    if question:
        with st.spinner("Generate response..."):
            st.session_state.messages.append({'role':'user', 'content':question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages, callbacks=[st_cb])

            st.session_state.messages.append({'role':'assistant', 'content':response})
            st.write('## Response:')
            st.success(response)

    else:
        st.warning("Please enter your question")