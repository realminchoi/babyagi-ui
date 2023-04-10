from collections import deque
from typing import Dict, List, Optional
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
import streamlit as st

class TaskCreationChain(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseLLM, objective: str, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_creation_template = """
            You are an task creation AI that uses the result of an execution agent
            to create new tasks with the following objective: {objective},
            The last completed task has the result: {result}.
            This result was based on this task description: {task_description}.
            These are incomplete tasks: {incomplete_tasks}.
            Based on the result, create new tasks to be completed
            by the AI system that do not overlap with incomplete tasks.
            Return the tasks as an array.
        """.strip()
        prompt = PromptTemplate(
            template=task_creation_template,
            partial_variables={"objective": objective},
            input_variables=["result", "task_description", "incomplete_tasks"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
    
    def get_next_task(self, result: Dict, task_description: str, task_list: List[str]) -> List[Dict]:
        """Get the next task."""
        incomplete_tasks = ", ".join(task_list)
        response = self.run(result=result, task_description=task_description, incomplete_tasks=incomplete_tasks)
        new_tasks = response.split('\n')
        return [{"task_name": task_name} for task_name in new_tasks if task_name.strip()]
    

class TaskPrioritizationChain(LLMChain):
    """Chain to prioritize tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, objective: str, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_prioritization_template = """
            You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing
            the following tasks: {task_names}.
            Consider the ultimate objective of your team: {objective}.
            Do not remove any tasks. Return the result as a numbered list, like:
            #. First task
            #. Second task
            Start the task list with number {next_task_id}."
        """.strip()
        prompt = PromptTemplate(
            template=task_prioritization_template,
            partial_variables={"objective": objective},
            input_variables=["task_names", "next_task_id"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

    def prioritize_tasks(self, this_task_id: int, task_list: List[Dict]) -> List[Dict]:
        """Prioritize tasks."""
        task_names = [t["task_name"] for t in task_list]
        next_task_id = int(this_task_id) + 1
        response = self.run(task_names=task_names, next_task_id=next_task_id)
        new_tasks = response.split('\n')
        prioritized_task_list = []
        for task_string in new_tasks:
            if not task_string.strip():
                continue
            task_parts = task_string.strip().split(".", 1)
            if len(task_parts) == 2:
                task_id = task_parts[0].strip()
                task_name = task_parts[1].strip()
                prioritized_task_list.append({"task_id": task_id, "task_name": task_name})
        return prioritized_task_list

        
class ExecutionChain(LLMChain):
    """Chain to execute tasks."""
    
    vectorstore: VectorStore = Field(init=False)

    @classmethod
    def from_llm(cls, llm: BaseLLM, vectorstore: VectorStore, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        execution_template = """
            You are an AI who performs one task based on the following objective: {objective}.
            Take into account these previously completed tasks: {context}.
            Your task: {task}.
            Response:
        """.strip()
        prompt = PromptTemplate(
            template=execution_template,
            input_variables=["objective", "context", "task"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose, vectorstore=vectorstore)
    
    def _get_top_tasks(self, query: str, k: int) -> List[str]:
        """Get the top k tasks based on the query."""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        if not results:
            return []
        sorted_results, _ = zip(*sorted(results, key=lambda x: x[1], reverse=True))
        return [str(item.metadata['task']) for item in sorted_results]
    
    def execute_task(self, objective: str, task: str, k: int = 5) -> str:
        """Execute a task."""
        context = self._get_top_tasks(query=objective, k=k)
        return self.run(objective=objective, context=context, task=task)


class Message:
    exp: st.expander
    ai_icon = "./img/robot.png"

    def __init__(self, label: str):
        message_area, icon_area = st.columns([10, 1])
        icon_area.image(self.ai_icon, caption="BabyAGI")

        # Expander
        self.exp = message_area.expander(label=label, expanded=True)

    def __enter__(self):
        return self

    def __exit__(self, ex_type, ex_value, trace):
        pass

    def write(self, content):
        self.exp.markdown(content)


class BabyAGI(BaseModel):
    """Controller model for the BabyAGI agent."""

    objective: str = Field(alias="objective")
    task_list: deque = Field(default_factory=deque)
    task_creation_chain: TaskCreationChain = Field(...)
    task_prioritization_chain: TaskPrioritizationChain = Field(...)
    execution_chain: ExecutionChain = Field(...)
    task_id_counter: int = Field(1)

    def add_task(self, task: Dict):
        self.task_list.append(task)

    def print_task_list(self):
        with Message(label="Task List") as m:
            m.write("### Task List")
            for t in self.task_list:
                m.write("- " + str(t["task_id"]) + ": " + t["task_name"])
                m.write("")

    def print_next_task(self, task: Dict):
        with Message(label="Next Task") as m:
            m.write("### Next Task")
            m.write("- " + str(task["task_id"]) + ": " + task["task_name"])
            m.write("")

    def print_task_result(self, result: str):
        with Message(label="Task Result") as m:
            m.write("### Task Result")
            m.write(result)
            m.write("")

    def print_task_ending(self):
        with Message(label="Task Ending") as m:
            m.write("### Task Ending")
            m.write("")


    def run(self, max_iterations: Optional[int] = None):
        """Run the agent."""
        num_iters = 0
        while True:
            if self.task_list:
                self.print_task_list()

                # Step 1: Pull the first task
                task = self.task_list.popleft()
                self.print_next_task(task)

                # Step 2: Execute the task
                result = self.execution_chain.execute_task(
                    self.objective, task["task_name"]
                )
                this_task_id = int(task["task_id"])
                self.print_task_result(result)

                # Step 3: Store the result in Pinecone
                result_id = f"result_{task['task_id']}"
                self.execution_chain.vectorstore.add_texts(
                    texts=[result],
                    metadatas=[{"task": task["task_name"]}],
                    ids=[result_id],
                )

                # Step 4: Create new tasks and reprioritize task list
                new_tasks = self.task_creation_chain.get_next_task(
                    result, task["task_name"], [t["task_name"] for t in self.task_list]
                )
                for new_task in new_tasks:
                    self.task_id_counter += 1
                    new_task.update({"task_id": self.task_id_counter})
                    self.add_task(new_task)
                self.task_list = deque(
                    self.task_prioritization_chain.prioritize_tasks(
                        this_task_id, list(self.task_list)
                    )
                )
            num_iters += 1
            if max_iterations is not None and num_iters == max_iterations:
                self.print_task_ending()
                break

    @classmethod
    def from_llm_and_objectives(
        cls,
        llm: BaseLLM,
        vectorstore: VectorStore,
        objective: str,
        first_task: str,
        verbose: bool = False,
    ) -> "BabyAGI":
        """Initialize the BabyAGI Controller."""
        task_creation_chain = TaskCreationChain.from_llm(
            llm, objective, verbose=verbose
        )
        task_prioritization_chain = TaskPrioritizationChain.from_llm(
            llm, objective, verbose=verbose
        )
        execution_chain = ExecutionChain.from_llm(llm, vectorstore, verbose=verbose)
        controller =  cls(
            objective=objective,
            task_creation_chain=task_creation_chain,
            task_prioritization_chain=task_prioritization_chain,
            execution_chain=execution_chain,
        )
        controller.add_task({"task_id": 1, "task_name": first_task})
        return controller


def main():
    st.set_page_config(
        initial_sidebar_state="expanded",
        page_title="BabyAGI UI",
        layout="centered",
    )

    with st.sidebar:
        openai_api_key = st.text_input('Your OpenAI API KEY', type="password")
        model_name = st.selectbox("Model name", options=["gpt-3.5-turbo", "gpt-4", "text-davinci-003"])
        temperature = st.slider(
            label="Temperature",
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            value=0.5,
        )

    st.title("BabyAGI UI")
    objective = st.text_input("Input Ultimate goal", "Solve world hunger")
    first_task = st.text_input("Input Where to start", "Develop a task list")
    max_iterations = st.number_input("Max iterations", value=3, min_value=1, step=1)
    button = st.button("Run")

    embedding_model = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(["_"], embedding_model, metadatas=[{"task":first_task}])

    if button:
        try:
            baby_agi = BabyAGI.from_llm_and_objectives(
                llm=OpenAI(openai_api_key=openai_api_key, temperature=temperature, model_name=model_name),
                vectorstore=vectorstore,
                objective=objective,
                first_task=first_task,
                verbose=False
            )
            baby_agi.run(max_iterations=max_iterations)
        except Exception as e:
            st.error(e)

if __name__ == "__main__":
    main()