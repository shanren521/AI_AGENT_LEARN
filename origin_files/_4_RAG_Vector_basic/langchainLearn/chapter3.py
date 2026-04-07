from langchain.schema import AIMessage,HumanMessage,SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain.chains.api.prompt import API_RESPONSE_PROMPT




"""
模型I/O
"""

# 聊天模型包装器
def chat_model_wrapper():
    chat = ChatOpenAI(model_name="gtp-4o", temperature=0.3)
    messages = [SystemMessage(content="你是个取名大师，擅长为创业公司取名字"),
                HumanMessage(content="帮我给新公司取个名字，是关于AI的")]
    response = chat(messages)
    print(response.content)
    

# 提示词模板的输入
def prompt_template_input():
    prompt = API_RESPONSE_PROMPT.format(api_docs="", question="", api_url="", api_response="")
    



    