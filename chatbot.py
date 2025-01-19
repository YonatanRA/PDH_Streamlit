# libraries

from tools import logger
from chapter_agent_tools import (ChapterAgent,
                                 get_chapter_1, 
                                 get_chapter_2,
                                 get_chapter_3,
                                 get_chapter_4,
                                 get_chapter_6
                                 )

from typing import Literal
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

import os
from dotenv import load_dotenv
load_dotenv()


# load enviroment variables
load_dotenv()

GLOBAL_PATH = os.path.dirname(os.path.abspath(__file__))

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

members = ['capitulo_1', 'capitulo_2', 'capitulo_3', 'capitulo_4', 'capitulo_6']


class Router(TypedDict):
    next: Literal[(*members,)] # type: ignore



class DesignHandbookBot:
    
    def __init__(self, thread_id: str):

        self.thread_id = thread_id
        self.memory = MemorySaver()

        self.base_model = ChatOpenAI(model_name='gpt-4o', temperature=0)


        chapter_1 = ChapterAgent('chapter_1_tool', get_chapter_1)
        chapter_1_tool_node = chapter_1.chapter_tool_node

        chapter_2 = ChapterAgent('chapter_2_tool', get_chapter_2)
        chapter_2_tool_node = chapter_2.chapter_tool_node

        chapter_3 = ChapterAgent('chapter_3_tool', get_chapter_3)
        chapter_3_tool_node = chapter_3.chapter_tool_node

        chapter_4 = ChapterAgent('chapter_4_tool', get_chapter_4)
        chapter_4_tool_node = chapter_4.chapter_tool_node

        chapter_6 = ChapterAgent('chapter_6_tool', get_chapter_6)
        chapter_6_tool_node = chapter_6.chapter_tool_node


        self.final_model = ChatOpenAI(model_name='gpt-4o', temperature=0)
        self.final_model = self.final_model.with_config(tags=['final_node'])


        # start builder
        builder = StateGraph(MessagesState)

        # nodes
        builder.add_node('chapter_supervisor', self.supervisor_agent)

        builder.add_node('capitulo_1', chapter_1.chapter_agent)
        builder.add_node('chapter_1_tool', chapter_1_tool_node)

        builder.add_node('capitulo_2', chapter_2.chapter_agent)
        builder.add_node('chapter_2_tool', chapter_2_tool_node)

        builder.add_node('capitulo_3', chapter_3.chapter_agent)
        builder.add_node('chapter_3_tool', chapter_3_tool_node)

        builder.add_node('capitulo_4', chapter_4.chapter_agent)
        builder.add_node('chapter_4_tool', chapter_4_tool_node)

        builder.add_node('capitulo_6', chapter_6.chapter_agent)
        builder.add_node('chapter_6_tool', chapter_6_tool_node)
        
        builder.add_node('final', self.final_agent)

        # edges
        builder.add_edge(START, 'chapter_supervisor')

        builder.add_edge('chapter_1_tool', 'capitulo_1')
        builder.add_edge('chapter_2_tool', 'capitulo_2')
        builder.add_edge('chapter_3_tool', 'capitulo_3')
        builder.add_edge('chapter_4_tool', 'capitulo_4')
        builder.add_edge('chapter_6_tool', 'capitulo_6')

        builder.add_conditional_edges('capitulo_1', chapter_1.chapter_continue)
        builder.add_conditional_edges('capitulo_2', chapter_2.chapter_continue)
        builder.add_conditional_edges('capitulo_3', chapter_3.chapter_continue)
        builder.add_conditional_edges('capitulo_4', chapter_4.chapter_continue)
        builder.add_conditional_edges('capitulo_6', chapter_6.chapter_continue)

        builder.add_edge('final', END)

        self.graph = builder.compile(checkpointer=self.memory)
        
    
    def supervisor_agent(self, state: MessagesState) -> Command[Literal[(*members,)]]:     # type: ignore

        system_prompt = f'''Eres un supervisor encargado de gestionar una conversación entre los 
                            siguientes trabajadores: {members}. Ten en cuenta lo siguiente:
                            
                            capitulo_1: Filosofía y principios de trabajo. 
                            Destaca el equipo como pilar central, promoviendo colaboración, 
                            crecimiento personal, y principios de trabajo fundamentales. Trata sobre el 
                            equipo en mendesaltaren, enfatizando la generosidad y honestidad como valores fundamentales. 
                            La organización se basa en squads para fomentar sinergias. 
                            Se prioriza el desarrollo personal y profesional, asegurando un crecimiento conjunto y continuo.

                            capitulo_2: Cultura.
                            Describe la cultura organizacional, principios de diseño, 
                            trabajo en equipo, y evolución hacia un entorno remoto y colaborativo. 
                            Destaca la cultura de mendesaltaren basada en valores compartidos, 
                            promoviendo un equipo de alto rendimiento y crecimiento constante. 
                            Utiliza dinámicas como summits y demos para fortalecer la identidad, 
                            adaptándose a cambios como el trabajo remoto y la expansión empresarial.

                            capitulo_3: Gestión y organización.
                            Describe una estructura organizativa antifrágil, basada en transparencia, 
                            flexibilidad y mejora continua, optimizando procesos y comunicación.
                            Se organiza en áreas productiva, operativa y satélites, optimizando procesos y 
                            fomentando la comunicación y colaboración interna y externa.

                            capitulo_4: Diseño estratégico.
                            Aborda el diseño estratégico, integrando innovación, investigación, 
                            ideación, producción y lanzamiento para conectar organizaciones con usuarios 
                            de manera efectiva. Trata el diseño estratégico como la sinergia entre innovación 
                            y pensamiento sistémico. Se enfoca en entender el "por qué" de las empresas 
                            para crear un "cómo" efectivo, priorizando un enfoque centrado en las personas y la mejora continua.

                            capitulo_6: Branding.
                            Aborda la creación de marcas con propósito, integrando branding y producto, 
                            mediante un enfoque holístico y estratégico. Se centra en crear marcas que trasciendan, 
                            combinando tecnología y creatividad. Se busca un branding que conecte con audiencias a 
                            través de una narrativa única y contemporánea, desarrollando marcas humanas y 
                            digitales con propósito y flexibilidad.

                            Dada la siguiente solicitud del usuario, 
                            responde con el trabajador que debe actuar a continuación. 
                            Cada trabajador realizará una tarea y responderá con sus resultados y estado. 
                            
                        '''
        

        n_last_messages = 8
    
        messages = [{'role': 'system', 'content': system_prompt},] + state['messages'][-n_last_messages:]

        logger.info(f'N memory messages in supervisor -- {len(messages)}')
    
        response = self.base_model.with_structured_output(Router).invoke(messages)
        
        goto = response['next']

        logger.info(f'SUPERVISOR -- {goto}')
        
        return Command(goto=goto)


    def final_agent(self, state: MessagesState):
    
        """
        Final model invoke for response
        """
        
        logger.info('Final Model')

        messages = state['messages']
        last_ai_message = messages[-1]
        
        response = self.final_model.invoke([SystemMessage('You are a repeater. Repeat the last message.'),
                                            HumanMessage(last_ai_message.content)])

        response.id = last_ai_message.id
        
        return {'messages': [response]}


    def invoke(self, question: str):

        config = {'configurable': {'thread_id': self.thread_id}}

        logger.info(f'Invoke in thread_id : {self.thread_id}')

        inputs = {'messages': [HumanMessage(content=question)]}


        for msg, metadata in self.graph.stream(inputs, config, stream_mode='messages'):
            if (
                msg.content
                and not isinstance(msg, HumanMessage)
                and metadata['langgraph_node'] == 'final'
            ):
                    yield(msg.content)

