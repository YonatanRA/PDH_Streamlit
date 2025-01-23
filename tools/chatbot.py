# libraries

from tools.tools import logger
from tools.chapter_agent_tools import ChapterAgent, chapter_tools

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

members = ['capitulo_1', 'capitulo_2', 'capitulo_3', 'capitulo_4', 'capitulo_6',
           'capitulo_11', 'capitulo_12', 'capitulo_13', 'capitulo_14', 'capitulo_15']


class Router(TypedDict):
    next: Literal[(*members,)] # type: ignore



class DesignHandbookBot:
    
    def __init__(self, thread_id: str):

        self.thread_id = thread_id
        self.memory = MemorySaver()

        self.base_model = ChatOpenAI(model='gpt-4o-mini', temperature=0)


        agents_tools = {m: {} for m in members}

        for m in members:
             agent = ChapterAgent(f'{m}_tool', chapter_tools[f'{m}_tool'])
             agents_tools[m]['agent'] = agent
             agents_tools[m]['tool_node'] = agent.chapter_tool_node


        self.final_model = ChatOpenAI(model_name='gpt-4o', temperature=0)
        self.final_model = self.final_model.with_config(tags=['final_node'])


        # start builder
        builder = StateGraph(MessagesState)


        # nodes
        builder.add_node('chapter_supervisor', self.supervisor_agent)

        for m in members:
            builder.add_node(m, agents_tools[m]['agent'].chapter_agent)
            builder.add_node(f'{m}_tool', agents_tools[m]['tool_node'])
             
        builder.add_node('final', self.final_agent)


        # edges
        builder.add_edge(START, 'chapter_supervisor')

        for m in members:
            builder.add_edge(f'{m}_tool', m)
            builder.add_conditional_edges(m, agents_tools[m]['agent'].chapter_continue)

        builder.add_edge('final', END)

        # compile graph and add memory
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


                            capitulo_11: Optimización de la tasa de conversión.
                            Trata sobre la Optimización de la Tasa de Conversión (CRO), 
                            utilizando análisis del comportamiento del usuario, metodología científica y 
                            trabajo multidisciplinar para mejorar conversiones y resultados de negocio en entornos digitales.


                            capitulo_12: Sistemas de diseño.
                            Aborda los sistemas de diseño, destacando su papel en la creación de patrones reutilizables 
                            que aseguran consistencia y eficiencia. Facilitan la colaboración entre diseño y desarrollo, 
                            permitiendo un lenguaje común y adaptabilidad en la evolución de productos digitales.


                            capitulo_13: Herramientas.
                            Detalla las herramientas y ejercicios utilizados en el proceso de diseño, 
                            explicando su propósito y cómo mejoran el trabajo diario. Incluye herramientas como Abstract, 
                            Sketch, Figma, Zeplin, Marvel, Notion y Asana.


                            capitulo_14: Product design, diseño de producto.
                            Detalla el procedimiento del diseñor de producto en mendesaltaren, 
                            destacando la importancia de un proceso bien estructurado. Incluye fases como preparación, 
                            comprensión, definición, producción y entrega, asegurando claridad, eficiencia y alineación con el 
                            cliente en cada etapa del proyecto.


                            capitulo_15: Design ops. Operaciones de diseño.
                            Aborda la gestión de versiones en sistemas de diseño, destacando la importancia de definir 
                            objetivos claros, realizar commits en Figma para registrar cambios y mantener un changelog. 
                            Se diferencia entre nueva versión, funcionalidad y corrección para asegurar consistencia.


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

