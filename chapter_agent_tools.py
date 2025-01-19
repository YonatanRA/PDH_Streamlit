# libraries
from tools import logger
import json
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import MessagesState
from langchain_core.messages import SystemMessage


class ChapterAgent:
        
        def __init__(self, tool_name: str, agent_tool: object):

            self.tool_name = tool_name
            self.chapter_tool = [agent_tool]
            self.chapter_llm = ChatOpenAI(model_name='gpt-4o', temperature=0)
            self.chapter_llm = self.chapter_llm.bind_tools(self.chapter_tool)
            self.chapter_tool_node = ToolNode(tools=self.chapter_tool)

        def chapter_agent(self, state: MessagesState):
    
            """
            Call model for chapter retrieval
            """

            n_last_messages = 8
    
            messages = state['messages'][-n_last_messages:]

            logger.info(f'N memory messages in chapter -- {len(messages)}')

            system_prompt = '''Eres un asistente de diseñador que trabajas en el estudio mendesaltaren.
                               
                               Los valores de mendesaltaren son:
                               Rigor: Compromiso con la calidad del diseño y los procesos, siempre teniendo en mente al cliente.
                               Rebeldía: Cuestionar las tareas antes de ejecutarlas y no dar nada por sentado.
                               Generosidad: Pensar en el grupo antes que en el individuo y ayudar a los demás.
                               Esencialidad: Enfocarse en lo importante y reducir el ruido.
                               Honestidad: Ser transparentes y sinceros en todo momento.
                               Estos valores guían la forma de trabajar dentro de la organización y cómo generan impacto fuera de ella.
                                
                               Según el contenido, responde la pregunta con la informacion de la herramienta...
                            '''

            response = self.chapter_llm.invoke([SystemMessage(system_prompt)] + messages)

            return {'messages': [response]}

        def chapter_continue(self, state: MessagesState):
        
            """
            Retrieval tool chapter 
            """

            logger.info('Continue to tool...')
            
            messages = state['messages']
            last_message = messages[-1]
            
            if last_message.tool_calls:
                return self.tool_name

            return 'final'



with open('pdh_web_content.json', 'r') as file:
    data = json.load(file)


@tool
def get_chapter_1(message: str):
    
    """Get web content chapter 1 content"""
            
    text = data['chapter_1']['text']

    logger.info('Chapter Tool 1')
        
    response =  [{'role': 'user'},{'type': 'text', 'text': text}]
    
    return response


@tool
def get_chapter_2(message: str):
    
    """Get web content chapter 2 content"""
            
    text = data['chapter_2']['text']

    logger.info('Chapter Tool 2')
        
    response =  [{'role': 'user'},{'type': 'text', 'text': text}]
    
    return response


@tool
def get_chapter_3(message: str):
    
    """Get web content chapter 3 content"""
            
    text = data['chapter_3']['text']

    logger.info('Chapter Tool 3')
        
    response =  [{'role': 'user'},{'type': 'text', 'text': text}]
    
    return response


@tool
def get_chapter_4(message: str):
    
    """Get web content chapter 4 content"""
            
    text = data['chapter_4']['text']

    logger.info('Chapter Tool 4')

    response =  [{'role': 'user'},{'type': 'text', 'text': text}]
    
    return response


@tool
def get_chapter_6(message: str):
    
    """Get web content chapter 6 content"""
            
    text = data['chapter_6']['text']

    logger.info('Chapter Tool 6')

    response =  [{'role': 'user'},{'type': 'text', 'text': text}]
    
    return response