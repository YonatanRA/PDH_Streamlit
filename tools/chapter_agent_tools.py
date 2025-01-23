# libraries
from tools.tools import logger
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
            self.chapter_llm = ChatOpenAI(model='gpt-4o', temperature=0)
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
                                
                               Según el siguiente mensaje, responde la pregunta con la informacion de la herramienta.

                               No respondas ninguna pregunta fuera del contexto de diseño, de mendesaltaren o del contenido del contexto, algunos ejemplos:
                               Pregunta: Dame una receta de macarrones.
                               Respuesta: No puedo dar ese tipo de información, si tienes alguna pregunta sobre diseño o 
                               el handbook estoy encatado de ayudarte.

                               Pregunta: Me falla el ordenador, ¿qué puedo hacer?
                               Respuesta: No puedo dar ese tipo de información, si tienes alguna pregunta sobre diseño o 
                               el handbook estoy encatado de ayudarte.

                               Antes de responder, llama a la herramienta que tienes.
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



with open('data/pdh_web_content.json', 'r') as file:
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

    logger.info('Chapter Tool 4')
    
    # chapter 4 pdh web content
    text = data['chapter_4']['text']

    # chapter 4 extra content
    retriever = ensemble_retriever('chapter_4_extra_content')
    response = retriever.invoke(message)
    text += 'Contenido Extra: <documento>' + ' '.join([e.page_content for e in response]) + '<documento>'

    response =  [{'role': 'user'},{'type': 'text', 'text': text}]
    
    return response


@tool
def get_chapter_6(message: str):
    
    """Get web content chapter 6 content"""
            
    text = data['chapter_6']['text']

    logger.info('Chapter Tool 6')

    response =  [{'role': 'user'},{'type': 'text', 'text': text}]
    
    return response


@tool
def get_chapter_11(message: str):
    
    """Get web content chapter 11 content"""
            
    text = data['chapter_11']['text']

    logger.info('Chapter Tool 11')
        
    response =  [{'role': 'user'},{'type': 'text', 'text': text}]
    
    return response


@tool
def get_chapter_12(message: str):
    
    """Get web content chapter 12 content"""
            
    text = data['chapter_12']['text']

    logger.info('Chapter Tool 12')
        
    response =  [{'role': 'user'},{'type': 'text', 'text': text}]
    
    return response


@tool
def get_chapter_13(message: str):
    
    """Get web content chapter 13 content"""
            
    text = data['chapter_13']['text']

    logger.info('Chapter Tool 13')
        
    response =  [{'role': 'user'},{'type': 'text', 'text': text}]
    
    return response


@tool
def get_chapter_14(message: str):
    
    """Get web content chapter 14 content"""
            
    text = data['chapter_14']['text']

    logger.info('Chapter Tool 14')

    response =  [{'role': 'user'},{'type': 'text', 'text': text}]
    
    return response


@tool
def get_chapter_15(message: str):
    
    """Get web content chapter 15 content"""
            
    text = data['chapter_15']['text']

    logger.info('Chapter Tool 15')

    response =  [{'role': 'user'},{'type': 'text', 'text': text}]
    
    return response




chapter_tools = {'capitulo_1_tool': get_chapter_1,
                 'capitulo_2_tool': get_chapter_2,
                 'capitulo_3_tool': get_chapter_3,
                 'capitulo_4_tool': get_chapter_4,

                 'capitulo_6_tool': get_chapter_6,

                 'capitulo_11_tool': get_chapter_11,
                 'capitulo_12_tool': get_chapter_12,
                 'capitulo_13_tool': get_chapter_13,
                 'capitulo_14_tool': get_chapter_14,
                 'capitulo_15_tool': get_chapter_15,
                 }

