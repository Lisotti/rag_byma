from setuptools import setup, find_packages

setup(
    name='rag_byma_project', # Dale un nombre único al paquete
    version='0.1.0',
    description='RAG Pipeline para análisis de BYMA y datos financieros',
    # Esto le dice a setuptools que busque paquetes en el directorio 'src'
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'langchain',
        'langchain-openai',
        'langchain-community',
        'chromadb',
        'pypdf',
        'gradio',
        'tiktoken',
        'cohere',
        'rank_bm25',
    ],
)