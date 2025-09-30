from langchain.chains import conversation
from langchain.chat_models import ChatOpenAI, PromptTemplate, LLMChain
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.base import ConversationChain
import os

from langchain_core.prompts import MessagesPlaceholder


# 代码1
def code1():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_api_key)
    _prompt = "Answer the question: {question}"
    prompt = PromptTemplate.from_template(_prompt)
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run({"question": "What is the capital of France?"})
    return {"response": response["text"]}


# 聊天机器人
def chat_robot():
    # temperature 0 表示确定性输出，1 表示随机性输出
    chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chat.predict_messages([
        HumanMessage(content="Translate this sentence from english to french. I love programming"),
    ])
    AIMessage(content="J'aime programmer", additional_kwargs={})


# 提示词模版
def template_prompt():
    template = "You are a helpful assistant that translates {input_language} to {output_language}."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming")


# 创建第一个链
def create_chain():
    # 初始化ChatOpenAI聊天模型，温度设置为0
    chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    # 定义系统消息模板
    template = "You are a helpful assistant that translates {input_language} to {output_language}."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    # 定义人类消息模板
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    # 将这两个模板组合到聊天提示词模板中
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    # 使用LLMChain组合聊天模型组件和提示词模板
    chain = LLMChain(llm=chat, prompt=chat_prompt)  
    # 运行链，传入参数
    chain.run(input_language="English", output_language="French", text="What do you do?")  
    
# SerpAPI Agent
def serpapi_agent():
    # 设置密钥
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["SERPAPI_API_KEY"] = ""
    
    # 加载大语言模型
    chat = ChatOpenAI(temerature=0)
    # 加载工具
    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    # 初始化agent
    agent = initialize_agent(tools, chat, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    # 测试agent
    agent.run("What will be the weather in Shanghai three days from now?")
     
# 记忆组件
def memory_component():
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("""The following is a friendly conversation between a human and an AI. 
                                                  The AI istalkative and provides lots of specific details from its context. 
                                                  If the AI does not know the answer to a question,it truthfully says it does not know."""),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template(f"(input)")
    ])
    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)
    conversation.predict(input="hello, i am terry")
    
    
    
if __name__ == "__main__":
    