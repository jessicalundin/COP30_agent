import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os
import tempfile
from dotenv import load_dotenv
import requests
from io import StringIO
import numpy as np
import time
from codecarbon import EmissionsTracker
import torch
import gc
import hashlib
import base64
import altair as alt

# Set tokenizers parallelism to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# RAG components
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Load environment variables
load_dotenv()

# Authentication credentials - ideally these would be stored in .env file
# or using a more secure method
DEFAULT_USERNAME = os.environ.get("APP_USERNAME")
DEFAULT_PASSWORD = os.environ.get("APP_PASSWORD")
DEFAULT_OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
DEFAULT_HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Function to create password hash
def make_password_hash(password):
    """Create a simple hash of the password"""
    return hashlib.sha256(password.encode()).hexdigest()

# Function to verify credentials
def check_credentials(username, password):
    """Check if username and password are correct"""
    correct_username = DEFAULT_USERNAME
    correct_password_hash = make_password_hash(DEFAULT_PASSWORD)
    
    return username == correct_username and make_password_hash(password) == correct_password_hash

# Add custom CSS to hide app elements when showing login screen
def inject_custom_css():
    hide_elements = """
        <style>
        .login-container {
            width: 100%;
            max-width: 400px;
            margin: 0 auto;
            padding: 40px;
            background-color: #f8f9fa;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .login-title {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
            text-align: center;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """
    st.markdown(hide_elements, unsafe_allow_html=True)

# Initialize session state variables for authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'failed_login' not in st.session_state:
    st.session_state.failed_login = False
if 'openai_key_from_login' not in st.session_state:
    st.session_state.openai_key_from_login = DEFAULT_OPENAI_KEY

# Login callback functions
def login():
    """Callback for login button"""
    if check_credentials(st.session_state.username, st.session_state.password):
        st.session_state.authenticated = True
        st.session_state.failed_login = False
        # Store the provided API key (if any)
        api_key = st.session_state.get('login_api_key', '')
        if api_key:
            st.session_state.openai_key_from_login = api_key
    else:
        st.session_state.authenticated = False
        st.session_state.failed_login = True

def logout():
    """Callback for logout button"""
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.password = ""
    st.session_state.login_api_key = ""

# Set page configuration
st.set_page_config(page_title="Climate & Health RAG Assistant", layout="wide", page_icon="🌍")

# Display login form if not authenticated
if not st.session_state.authenticated:
    inject_custom_css()
    
    st.markdown("<h1 style='text-align: center;'>Climate & Health RAG Assistant</h1>", unsafe_allow_html=True)
    
    # Center the login content
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<div class='login-container'>", unsafe_allow_html=True)
        st.markdown("<div class='login-title'>Login Required</div>", unsafe_allow_html=True)
        
        # Display error message if login failed
        if st.session_state.failed_login:
            st.error("Invalid username or password. Please try again.")
        
        # Login form
        st.text_input("Username", key="username", placeholder="Enter your username")
        st.text_input("Password", type="password", key="password", placeholder="Enter your password")
        
        # Optional OpenAI API key input
        st.text_input("OpenAI API Key (Optional)", type="password", key="login_api_key", 
                     placeholder="Enter your OpenAI API key if needed")
        
        st.button("Login", on_click=login)
        
        st.markdown("<p style='text-align: center; margin-top: 20px; font-size: 12px;'>Contact the administrator if you need access.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Stop execution of the rest of the app
    st.stop()

# If we're here, the user is authenticated
# Set the API key from login if provided
if st.session_state.openai_key_from_login:
    os.environ["OPENAI_API_KEY"] = st.session_state.openai_key_from_login

# Sidebar for model selection and API key input
with st.sidebar:
    # Add logout button at the top of the sidebar
    st.button("Logout", on_click=logout)
    
    st.title("Settings")
    
    model_option = st.radio(
        "Select LLM Model:",
        ["OpenAI (GPT-3.5)", "Llama 3.2 (1B)", "Llama 3.2 (3B)"]
    )
    
    # Only ask for API key if OpenAI is selected and not provided at login
    openai_api_key = st.session_state.openai_key_from_login
    if model_option == "OpenAI (GPT-3.5)" and not openai_api_key:
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # Display emissions tracker options
    st.markdown("---")
    st.markdown("## Carbon Emissions")
    show_emissions = st.checkbox("Track carbon emissions for queries", value=True)
    
    # Information about models and emissions
    st.markdown("---")
    st.markdown("## About")
    st.markdown("This app allows you to ask questions about Climate Change, COP30, Brazil, and Dengue Fever using a Retrieval-Augmented Generation (RAG) system.")
    st.markdown("You can also explore climate data visualizations.")
    
    if model_option != "OpenAI (GPT-3.5)":
        st.info("Using Llama models runs computations locally on your device, which may use more power but protects privacy and doesn't require an API key.")

# Global variables for Llama models
llama_model = None
llama_tokenizer = None

# Function to load Llama model
@st.cache_resource
def load_llama_model(model_size="1B"):
    """Load the Llama 3.2 model based on size selection"""
    with st.spinner(f"Loading Llama 3.2 {model_size} model. This might take a minute..."):
        # Clear GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        model_id = f"meta-llama/Llama-3.2-{model_size}-Instruct"
        
        # Configure 4-bit quantization for memory efficiency
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.float16,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        # )
        
        # Load model with reduced precision
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            # quantization_config=bnb_config,
            torch_dtype=torch.float16,
            token=DEFAULT_HF_TOKEN
        )
        
        return model, tokenizer

# Function to create a carbon tracker
def create_tracker(model_name):
    """Create a carbon emissions tracker"""
    return EmissionsTracker(
        project_name=f"climate-rag-{model_name}",
        output_dir=".",
        log_level="warning",
        save_to_file=False,
        tracking_mode="process"
    )

# Function to estimate carbon emissions for OpenAI queries
def estimate_openai_emissions(question_length, answer_length):
    """
    Estimate carbon emissions for OpenAI API queries
    Based on research estimates for cloud-based LLM inference
    """
    # Approximate emission factors in gCO2eq per token
    # These are simplified estimates based on research
    per_token_emission = 0.000001  # 0.001 mgCO2eq per token (simplified estimate)
    total_tokens = question_length + answer_length
    return total_tokens * per_token_emission

# Function to create a RAG system from documents
@st.cache_resource(show_spinner=False)
def create_rag_system(docs_data, model_option="OpenAI (GPT-3.5)"):
    with st.spinner("Processing documents... This might take a minute."):
        temp_dir = tempfile.mkdtemp()
        document_loaders = []
        
        # Process uploaded documents
        for i, doc_data in enumerate(docs_data):
            file_path = os.path.join(temp_dir, f"document_{i}.pdf")
            with open(file_path, "wb") as f:
                f.write(doc_data)
            document_loaders.append(PyPDFLoader(file_path))
        
        # Load documents from the docs directory
        docs_dir = "docs"
        if os.path.exists(docs_dir):
            for filename in os.listdir(docs_dir):
                file_path = os.path.join(docs_dir, filename)
                if filename.lower().endswith('.pdf'):
                    document_loaders.append(PyPDFLoader(file_path))
                elif filename.lower().endswith(('.txt', '.md')):
                    document_loaders.append(TextLoader(file_path))
        else:
            st.warning("Docs directory not found. Create a 'docs' folder and add your documents there.")
        
        # Load and process documents
        documents = []
        for loader in document_loaders:
            try:
                documents.extend(loader.load())
            except Exception as e:
                st.warning(f"Error loading document: {e}")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)
        
        # Select embeddings model based on availability
        if model_option == "OpenAI (GPT-3.5)" and os.environ.get("OPENAI_API_KEY"):
            embeddings = OpenAIEmbeddings()
        else:
            # Use sentence-transformers for embeddings regardless of the LLM choice
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create vector store
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Create QA chain with appropriate LLM
        if model_option == "OpenAI (GPT-3.5)" and os.environ.get("OPENAI_API_KEY"):
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        else:
            # For Llama models, we'll use our custom implementation
            # But set up a placeholder HuggingFace endpoint
            llm = HuggingFaceEndpoint(
                endpoint_url="local_llama",
                task="text-generation",
                max_new_tokens=512,
                temperature=0.1,
                model_kwargs={},
                token=DEFAULT_HF_TOKEN
            )
            
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        return qa_chain, vectorstore, retriever

# Function to run Llama inference
def run_llama_inference(model, tokenizer, prompt, max_new_tokens=512):
    """Generate a response using the Llama model"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.95,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the response
    response = response[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]
    return response.strip()

# Function to get sample text about COP30
def get_COP30_info():
    return """
    COP30: The 29th Conference of the Parties to the UN Framework Convention on Climate Change
    
    COP30 is scheduled to be held in Baku, Azerbaijan in November 2024. This conference is a crucial part of the global effort to combat climate change, following the Paris Agreement's framework.
    
    Key topics expected at COP30:
    1. Climate finance - particularly focusing on the new climate finance goal beyond the previous $100 billion annual target
    2. Implementation of the Loss and Damage Fund established at COP27
    3. Assessment of progress on nationally determined contributions (NDCs)
    4. Advancing adaptation measures for vulnerable countries
    5. Strengthening mitigation efforts to limit global warming to 1.5°C
    
    COP30 comes at a critical time when global emissions need to be reduced drastically to meet the Paris Agreement goals. The conference will address the gap between current pledges and what science indicates is necessary to avoid catastrophic climate impacts.
    
    The previous conference, COP28 in Dubai, UAE, concluded with agreements on the operationalization of the loss and damage fund and the first global stocktake of the Paris Agreement. However, many climate advocates noted that more ambitious action is still needed on fossil fuel phase-out and adaptation financing.
    """

# Function to get sample text about Dengue Fever and climate change
def get_dengue_info():
    return """
    Dengue Fever and Climate Change: Impacts on Brazil and Global Health
    
    Dengue fever is a mosquito-borne viral infection that causes flu-like illness, and occasionally develops into a potentially lethal complication called severe dengue. The incidence of dengue has grown dramatically around the world in recent decades, with Brazil being one of the most affected countries.
    
    Climate Connection:
    - Rising temperatures extend the geographic range of Aedes mosquitoes that transmit dengue
    - Increased rainfall creates more breeding sites for mosquitoes
    - Longer warm seasons extend the transmission period
    - Urban heat islands in cities amplify these effects
    
    Brazil and Dengue:
    Brazil has experienced some of the largest dengue outbreaks globally. In 2023-2024, Brazil faced one of its worst dengue epidemics in history, with over 4 million cases reported in the first five months of 2024 alone.
    
    Climate change is expanding the dengue risk zone in Brazil, with cases now appearing in previously unaffected southern regions. Studies predict that by 2050, under high-emission scenarios, dengue risk areas in Brazil could expand by 45-100%.
    
    Public health responses include:
    - Mosquito control programs
    - Early warning systems based on climate forecasting
    - Vaccination campaigns
    - Community education
    
    The intersection of climate policy and public health policy is increasingly important for addressing vector-borne diseases like dengue fever, highlighting the need for integrated approaches at events like COP30.
    """

# Function to get climate data
@st.cache_data
def get_climate_data():
    # Global temperature anomalies (NASA GISS data)
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v3/GLB.Ts+dSST.csv"
    # response = requests.get(url)
    data = pd.read_csv(url, skiprows=1)
    data = data.iloc[:-1]  # Remove the last row which typically contains notes
    # data = data.replace(["***", ""], np.nan, inplace=True)
    
    # Clean up data
    data['Year'] = data['Year'].astype(int)
    
    # Convert all numeric columns to float, replacing non-numeric values with NaN
    for col in data.columns[1:13]:  # Skip the 'Year' column
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Create annual averages
    data['Annual'] = data.iloc[:, 1:13].mean(axis=1)
    
    # Select relevant columns
    global_temp = data[['Year', 'Annual']]
    global_temp = global_temp.rename(columns={'Annual': 'Temperature_Anomaly'})
    
    # Return cleaned dataset
    return global_temp

# Function to get Brazil climate data
@st.cache_data
def get_brazil_climate_data():
    # This is simulated data based on general trends for Brazil
    # In a real application, you would fetch actual data from a reputable source
    years = range(1990, 2024)
    
    # Simulated temperature anomalies for Brazil (generally following global trend but with variations)
    temp_anomalies = [0.15, 0.25, 0.08, 0.13, 0.21, 0.35, 0.22, 0.46, 0.52, 0.31, 
                       0.37, 0.48, 0.56, 0.53, 0.48, 0.63, 0.52, 0.57, 0.49, 0.60,
                       0.67, 0.54, 0.69, 0.72, 0.76, 0.87, 0.92, 0.84, 0.91, 0.97,
                       1.02, 0.94, 1.16, 1.24]
    
    # Simulated rainfall change percentage (more variable)
    rainfall_change = [-2, -1, -3, 1, -4, -5, -2, 3, -7, -4,
                        -6, -3, -8, -5, -7, -9, -4, -8, -12, -6,
                        -9, -5, -11, -8, -10, -13, -7, -14, -9, -15,
                        -12, -16, -10, -18]
    
    # Simulated dengue cases (increasing trend with year-to-year variation)
    dengue_cases = [460000, 428000, 490000, 510000, 580000, 540000, 620000, 720000, 680000, 740000,
                     850000, 810000, 890000, 930000, 980000, 1120000, 1050000, 1180000, 1240000, 1350000,
                     1280000, 1410000, 1560000, 1620000, 1730000, 1840000, 1950000, 2080000, 2240000, 2150000,
                     2350000, 2460000, 4120000, 4580000]
    
    # Create dataframe
    brazil_data = pd.DataFrame({
        'Year': years,
        'Temperature_Anomaly': temp_anomalies,
        'Rainfall_Change_Percent': rainfall_change,
        'Dengue_Cases': dengue_cases
    })
    
    return brazil_data

# Main app layout
st.title("🌍 Climate & Health Interactive Assistant")

# Create tabs
tab1, tab2, tab3 = st.tabs(["RAG Assistant", "Climate Data", "Brazil & Dengue"])

# Tab 1: RAG Assistant
with tab1:
    st.header("Ask me about Climate Change, COP30, Brazil, or Dengue Fever")
    
    # Document uploader
    st.subheader("Upload Documents (Optional)")
    uploaded_files = st.file_uploader("Upload PDF documents for additional context", type="pdf", accept_multiple_files=True)
    
    # Process uploaded documents
    docs_data = []
    if uploaded_files:
        for file in uploaded_files:
            docs_data.append(file.read())
    
    # Initialize RAG system and model based on selection
    qa_chain = None
    vectorstore = None
    retriever = None
    
    # Load Llama model if selected
    if model_option in ["Llama 3.2 (1B)", "Llama 3.2 (3B)"]:
        model_size = "1B" if model_option == "Llama 3.2 (1B)" else "3B"
        llama_model, llama_tokenizer = load_llama_model(model_size)
    
    # Create RAG system if conditions are met
    valid_setup = (
        (model_option == "OpenAI (GPT-3.5)" and openai_api_key) or
        model_option in ["Llama 3.2 (1B)", "Llama 3.2 (3B)"]
    )
    
    if valid_setup and (docs_data or not uploaded_files):
        with st.spinner("Setting up the knowledge base..."):
            qa_chain, vectorstore, retriever = create_rag_system(docs_data, model_option)
    
    # Question input
    st.subheader("Ask a Question")
    question = st.text_input("Type your question here:", placeholder="e.g., What are the main issues to be discussed at COP30?")
    
    if st.button("Get Answer"):
        if model_option == "OpenAI (GPT-3.5)" and not openai_api_key:
            st.error("Please enter your OpenAI API key in the sidebar to use GPT-3.5.")
        elif not question:
            st.warning("Please enter a question.")
        elif not valid_setup:
            st.error("Model setup is not valid. Please check your settings.")
        else:
            # Create an emissions tracker container
            emissions_container = st.container()
            
            with st.spinner("Searching for information..."):
                try:
                    # Start emissions tracking if enabled
                    emissions = 0
                    start_time = time.time()
                    
                    if model_option == "OpenAI (GPT-3.5)":
                        # Use OpenAI via LangChain
                        result = qa_chain({"query": question})
                        answer = result["result"]
                        source_documents = result["source_documents"]
                        
                        # Estimate emissions for OpenAI
                        if show_emissions:
                            question_tokens = len(question.split())
                            answer_tokens = len(answer.split())
                            emissions = estimate_openai_emissions(question_tokens, answer_tokens)
                            
                    else:
                        # Use Llama models directly
                        # First get context from retriever
                        retrieved_docs = retriever.invoke({"query": question})
                        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                        
                        # Create prompt with context
                        prompt = f"""Below is some context information:
                        
{context}

Based on the above context, please answer the following question:
Question: {question}

Answer:"""
                        
                        # Start emissions tracker
                        tracker = None
                        if show_emissions:
                            tracker = create_tracker(model_option)
                            tracker.start()
                            
                        # Generate response with Llama
                        answer = run_llama_inference(llama_model, llama_tokenizer, prompt)
                        source_documents = retrieved_docs
                        
                        # Stop emissions tracker
                        if show_emissions and tracker:
                            emissions = tracker.stop()
                    
                    # Calculate time taken
                    end_time = time.time()
                    time_taken = end_time - start_time
                    
                    # Display answer
                    st.subheader("Answer")
                    st.write(answer)
                    
                    # Display sources
                    st.subheader("Sources")
                    for i, doc in enumerate(source_documents):
                        with st.expander(f"Source {i+1}"):
                            st.write(doc.page_content)
                            st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
                    
                    # Display emissions information
                    if show_emissions:
                        with emissions_container:
                            st.markdown("---")
                            st.subheader("💨 Carbon Footprint")
                            
                            if model_option == "OpenAI (GPT-3.5)":
                                st.write(f"⏱️ Query time: {time_taken:.2f} seconds")
                                st.write(f"🔄 Estimated carbon emissions: {emissions*1000:.4f} mg CO₂e")
                                st.caption("Cloud API emissions are estimates based on research literature")
                            else:
                                st.write(f"⏱️ Query time: {time_taken:.2f} seconds")
                                st.write(f"🔄 Carbon emissions: {emissions*1000:.4f} mg CO₂e")
                                st.caption("Local emissions are measured based on your device's power consumption")
                            
                            # Comparison to everyday activities
                            st.markdown("### Equivalent to:")
                            breathing_seconds = emissions / 0.0000001  # CO2 from breathing (very rough estimate)
                            car_meters = emissions / 0.000193  # CO2 from car per meter (rough estimate)
                            
                            st.write(f"🚶 Breathing for {breathing_seconds:.2f} seconds")
                            st.write(f"🚗 Driving a car for {car_meters:.2f} meters")
                            
                            # Cumulative impact visualization
                            if 'total_emissions' not in st.session_state:
                                st.session_state.total_emissions = 0
                                st.session_state.query_count = 0
                            
                            st.session_state.total_emissions += emissions
                            st.session_state.query_count += 1
                            
                            st.write(f"📊 Your queries today: {st.session_state.query_count}")
                            st.write(f"📊 Total emissions today: {st.session_state.total_emissions*1000:.4f} mg CO₂e")
                            
                            # Comparative model footprint
                            st.markdown("### Model Comparison (estimated per query):")
                            models_data = {
                                "Model": ["GPT-3.5", "Llama 3.2 (1B)", "Llama 3.2 (3B)", "GPT-4"],
                                "CO₂e (mg)": [0.5, 0.2, 0.4, 2.1]  # Simplified comparative estimates
                            }
                            df_models = pd.DataFrame(models_data)
                            
                            fig = px.bar(df_models, x="Model", y="CO₂e (mg)", 
                                        title="Carbon Footprint by Model",
                                        color="Model", 
                                        color_discrete_sequence=px.colors.qualitative.Pastel)
                            st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    if "API key" in str(e):
                        st.warning("Please check that your OpenAI API key is valid.")

# Tab 2: Climate Data
with tab2:
    st.header("Global Climate Data Visualization")
    
    # Load data
    with st.spinner("Loading climate data..."):
        global_temp = get_climate_data()
    
    st.subheader("Global Temperature Anomalies (1880-Present)")
    
    # Create visualization options
    viz_type = st.radio("Select visualization type:", ["Line Chart", "Bar Chart", "Interactive Plot"])
    
    # Filter for a specific date range
    year_range = st.slider("Select Year Range:", 
                           min_value=int(global_temp['Year'].min()), 
                           max_value=int(global_temp['Year'].max()), 
                           value=(1950, int(global_temp['Year'].max())))
    
    # Filter data based on selected years
    filtered_data = global_temp[(global_temp['Year'] >= year_range[0]) & (global_temp['Year'] <= year_range[1])]
    
    # Display visualizations based on selection
    if viz_type == "Line Chart":
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(filtered_data['Year'], filtered_data['Temperature_Anomaly'], marker='o', linewidth=2)
        ax.set_title('Global Temperature Anomalies')
        ax.set_xlabel('Year')
        ax.set_ylabel('Temperature Anomaly (°C)')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(filtered_data['Year'], filtered_data['Temperature_Anomaly'], 1)
        p = np.poly1d(z)
        ax.plot(filtered_data['Year'], p(filtered_data['Year']), "r--", alpha=0.8)
        
        st.pyplot(fig)
        
    elif viz_type == "Bar Chart":
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(filtered_data['Year'], filtered_data['Temperature_Anomaly'], 
                      width=0.8, alpha=0.7)
        
        # Color bars based on temperature anomaly value
        for i, bar in enumerate(bars):
            if filtered_data['Temperature_Anomaly'].iloc[i] <= 0:
                bar.set_color('skyblue')
            else:
                bar.set_color('coral')
                
        ax.set_title('Global Temperature Anomalies')
        ax.set_xlabel('Year')
        ax.set_ylabel('Temperature Anomaly (°C)')
        ax.grid(True, alpha=0.3, axis='y')
        
        st.pyplot(fig)
        
    else:  # Interactive Plot
        fig = px.line(filtered_data, x='Year', y='Temperature_Anomaly', 
                      title='Global Temperature Anomalies',
                      labels={'Temperature_Anomaly': 'Temperature Anomaly (°C)'},
                      markers=True)
        
        # Add trend line
        fig.add_traces(
            px.scatter(filtered_data, x='Year', y='Temperature_Anomaly', 
                       trendline='ols').data[1]
        )
        
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    
    # Add some analysis
    st.subheader("Analysis")
    
    avg_recent = global_temp[global_temp['Year'] >= 2010]['Temperature_Anomaly'].mean()
    avg_baseline = global_temp[(global_temp['Year'] >= 1951) & (global_temp['Year'] <= 1980)]['Temperature_Anomaly'].mean()
    
    st.write(f"Average temperature anomaly since 2010: **{avg_recent:.2f}°C**")
    st.write(f"Average temperature anomaly during baseline period (1951-1980): **{avg_baseline:.2f}°C**")
    st.write(f"Increase: **{avg_recent - avg_baseline:.2f}°C**")
    
    st.info("The data shows a clear warming trend over time, with recent decades showing significantly higher temperature anomalies compared to the baseline period. This aligns with scientific consensus on global warming and climate change.")

# Tab 3: Brazil & Dengue
with tab3:
    """
    Function to display the Brazil and Dengue tab content
    """
    # App title and description
    st.header("Dengue Cases in Brazil: Two-Decade Comparison")
    st.markdown("""
    This visualization shows the significant increase in dengue cases in Brazil over three time periods:
    - January 2000 to June 2004
    - January 2010 to June 2014
    - January 2020 to June 2024
    """)

    # Create dataframe with the dengue cases data
    data = {
        'Period': ['Jan 2000 - Jun 2004', 'Jan 2010 - Jun 2014', 'Jan 2020 - Jun 2024'],
        'Probable Cases': [2073194, 6260684, 11236426],
        'Start Year': [2000, 2010, 2020],
        'Years': ['2000-2004', '2010-2014', '2020-2024']
    }

    df = pd.DataFrame(data)

    # Create visualization select box
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Bar Chart", "Line Chart", "Area Chart", "Comparative Analysis"]
    )

    # Display raw data if requested
    if st.checkbox("Show Raw Data"):
        st.subheader("Raw Data")
        st.dataframe(df)

    # Calculate additional statistics
    total_cases = df['Probable Cases'].sum()
    percent_increase_first_decade = ((df['Probable Cases'][1] - df['Probable Cases'][0]) / df['Probable Cases'][0]) * 100
    percent_increase_second_decade = ((df['Probable Cases'][2] - df['Probable Cases'][1]) / df['Probable Cases'][1]) * 100
    percent_increase_total = ((df['Probable Cases'][2] - df['Probable Cases'][0]) / df['Probable Cases'][0]) * 100

    # Main visualization area
    st.subheader(f"Visualization: {viz_type}")

    if viz_type == "Bar Chart":
        # Creating the bar chart with Altair for better interactivity
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('Period:N', title='Time Period', sort=None),
            y=alt.Y('Probable Cases:Q', title='Number of Probable Dengue Cases'),
            color=alt.Color('Period:N', legend=None),
            tooltip=['Period', 'Probable Cases']
        ).properties(
            height=400
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)

    elif viz_type == "Line Chart":
        # Create a line chart
        chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X('Start Year:Q', title='Starting Year'),
            y=alt.Y('Probable Cases:Q', title='Number of Probable Dengue Cases'),
            tooltip=['Period', 'Probable Cases']
        ).properties(
            height=400
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)

    elif viz_type == "Area Chart":
        # Create an area chart
        chart = alt.Chart(df).mark_area().encode(
            x=alt.X('Period:N', title='Time Period', sort=None),
            y=alt.Y('Probable Cases:Q', title='Number of Probable Dengue Cases'),
            color=alt.Color('Period:N', legend=None),
            tooltip=['Period', 'Probable Cases']
        ).properties(
            height=400
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)

    elif viz_type == "Comparative Analysis":
        # Create a more detailed analysis with multiple charts
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Total Cases by Period")
            chart1 = alt.Chart(df).mark_bar().encode(
                x=alt.X('Period:N', title='Time Period', sort=None),
                y=alt.Y('Probable Cases:Q', title='Number of Probable Cases'),
                color=alt.Color('Period:N', legend=None),
                tooltip=['Period', 'Probable Cases']
            ).properties(
                height=300
            ).interactive()
            
            st.altair_chart(chart1, use_container_width=True)
        
        with col2:
            st.subheader("Growth Trend")
            chart2 = alt.Chart(df).mark_line(point=True).encode(
                x=alt.X('Start Year:Q', title='Starting Year'),
                y=alt.Y('Probable Cases:Q', title='Number of Probable Cases'),
                tooltip=['Period', 'Probable Cases']
            ).properties(
                height=300
            ).interactive()
            
            st.altair_chart(chart2, use_container_width=True)
        
        # Create percentage increase visualization
        st.subheader("Percentage Increase Between Periods")
        
        increase_data = {
            'Comparison': ['2000-2004 vs 2010-2014', '2010-2014 vs 2020-2024', 'Total (2000-2004 vs 2020-2024)'],
            'Percentage Increase': [percent_increase_first_decade, percent_increase_second_decade, percent_increase_total]
        }
        
        increase_df = pd.DataFrame(increase_data)
        
        chart3 = alt.Chart(increase_df).mark_bar().encode(
            x=alt.X('Comparison:N', title='Comparison Period'),
            y=alt.Y('Percentage Increase:Q', title='Percentage Increase (%)'),
            color=alt.Color('Comparison:N', legend=None),
            tooltip=['Comparison', 'Percentage Increase']
        ).properties(
            height=300
        ).interactive()
        
        st.altair_chart(chart3, use_container_width=True)

    # Key metrics display
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Total Dengue Cases (2000-2024)",
            value=f"{total_cases:,}"
        )

    with col2:
        st.metric(
            label="Increase (2000-2004 to 2010-2014)",
            value=f"{percent_increase_first_decade:.1f}%",
            delta=f"{df['Probable Cases'][1] - df['Probable Cases'][0]:,} cases"
        )

    with col3:
        st.metric(
            label="Increase (2010-2014 to 2020-2024)",
            value=f"{percent_increase_second_decade:.1f}%",
            delta=f"{df['Probable Cases'][2] - df['Probable Cases'][1]:,} cases"
        )

    # Additional insights
    st.subheader("Key Insights")
    st.markdown(f"""
    - From 2000-2004 to 2020-2024, there was a **{percent_increase_total:.1f}%** increase in probable dengue cases in Brazil.
    - The total number of reported probable dengue cases across all three periods was **{total_cases:,}**.
    - The most recent period (2020-2024) accounts for **{(df['Probable Cases'][2]/total_cases*100):.1f}%** of all dengue cases in the recorded periods.
    - Between 2010-2014 and 2020-2024, there was an increase of **{df['Probable Cases'][2] - df['Probable Cases'][1]:,}** cases.
    """)

    # Add download functionality for the data
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name="dengue_cases_brazil.csv",
        mime="text/csv",
    )
    # st.header("Brazil Climate and Dengue Fever")
    
    # # Load Brazil-specific data
    # brazil_data = get_brazil_climate_data()
    
    # # Visualization selector
    # viz_option = st.selectbox(
    #     "Select data to visualize:",
    #     ["Temperature and Dengue Cases", "Rainfall Changes", "Dengue Cases by Year"]
    # )
    
    # if viz_option == "Temperature and Dengue Cases":
    #     st.subheader("Temperature Anomalies and Dengue Cases in Brazil")
        
    #     # Create two y-axes
    #     fig, ax1 = plt.subplots(figsize=(12, 6))
        
    #     color = 'tab:red'
    #     ax1.set_xlabel('Year')
    #     ax1.set_ylabel('Temperature Anomaly (°C)', color=color)
    #     ax1.plot(brazil_data['Year'], brazil_data['Temperature_Anomaly'], color=color, marker='o')
    #     ax1.tick_params(axis='y', labelcolor=color)
        
    #     ax2 = ax1.twinx()
    #     color = 'tab:blue'
    #     ax2.set_ylabel('Dengue Cases', color=color)
    #     ax2.plot(brazil_data['Year'], brazil_data['Dengue_Cases'], color=color, marker='s')
    #     ax2.tick_params(axis='y', labelcolor=color)
        
    #     fig.tight_layout()
    #     plt.title('Temperature Anomalies and Dengue Cases in Brazil')
    #     plt.grid(True, alpha=0.3)
        
    #     st.pyplot(fig)
        
    #     st.write("This visualization shows the relationship between rising temperatures and dengue cases in Brazil. As temperatures have increased, there has been a corresponding rise in dengue cases, particularly accelerating in recent years.")
        
    # elif viz_option == "Rainfall Changes":
    #     st.subheader("Annual Rainfall Changes in Brazil")
        
    #     fig, ax = plt.subplots(figsize=(12, 6))
    #     bars = ax.bar(brazil_data['Year'], brazil_data['Rainfall_Change_Percent'], width=0.7)
        
    #     # Color bars based on values
    #     for i, bar in enumerate(bars):
    #         if brazil_data['Rainfall_Change_Percent'].iloc[i] < 0:
    #             bar.set_color('brown')
    #         else:
    #             bar.set_color('green')
                
    #     ax.set_title('Annual Rainfall Change in Brazil (%)')
    #     ax.set_xlabel('Year')
    #     ax.set_ylabel('Rainfall Change (%)')
    #     ax.grid(True, alpha=0.3, axis='y')
        
    #     st.pyplot(fig)
        
    #     st.write("Brazil has experienced an overall trend of decreased rainfall in many regions, which affects water availability for communities, agriculture, and ecosystems. This pattern is consistent with climate change projections for parts of Brazil.")
        
    # else:  # Dengue Cases by Year
    #     st.subheader("Dengue Fever Cases in Brazil by Year")
        
    #     fig = px.bar(brazil_data, x='Year', y='Dengue_Cases',
    #                 title='Annual Dengue Cases in Brazil',
    #                 labels={'Dengue_Cases': 'Number of Cases'},
    #                 color='Dengue_Cases',
    #                 color_continuous_scale='YlOrRd')
        
    #     fig.update_layout(hovermode="x unified")
    #     st.plotly_chart(fig, use_container_width=True)
        
    #     # Calculate statistics
    #     recent_average = brazil_data[brazil_data['Year'] >= 2020]['Dengue_Cases'].mean()
    #     early_average = brazil_data[brazil_data['Year'] < 2000]['Dengue_Cases'].mean()
    #     percent_increase = ((recent_average - early_average) / early_average) * 100
        
    #     st.metric("Average Annual Cases (2020-2023)", f"{int(recent_average):,}", 
    #              f"{percent_increase:.1f}% since 1990s")
        
    #     st.write("Dengue cases in Brazil have increased dramatically over the past three decades. The most recent years show an unprecedented surge, with over 4 million cases reported in early 2024 alone.")
        
    #     st.info("Climate factors contributing to increased dengue transmission include: higher temperatures that extend the mosquito season, changes in precipitation patterns creating more breeding sites, and urbanization creating heat islands that further amplify these effects.")
        
    #     # COP30 relevance
    #     st.subheader("Relevance to COP30")
    #     st.write("Vector-borne diseases like dengue represent a critical public health impact of climate change. At COP30, health adaptation measures and funding for climate-sensitive disease control will be important topics, especially for vulnerable countries like Brazil.")

# Footer
st.markdown("---")
st.caption("Data sources: NASA GISS Surface Temperature Analysis, simulated Brazil climate and dengue data based on trends from scientific literature.")

# Carbon footprint of the app itself
with st.expander("💨 About Carbon Footprint Tracking"):
    st.markdown("""
    ### Carbon Emissions Information
    
    This application tracks the carbon emissions of your queries to raise awareness about the environmental impact of AI and machine learning models.
    
    #### Model Comparisons:
    - **OpenAI GPT models** run on cloud servers and have emissions associated with data center energy use
    - **Llama 3.2 models** run locally on your device, with emissions based on your computer's power consumption
    
    #### Why Track Emissions?
    AI and machine learning have significant and growing carbon footprints. By making these impacts visible, we can:
    - Make more informed choices about which models to use
    - Consider the environmental impact of our technology choices
    - Support the development of more efficient AI systems
    
    #### Methodology:
    - OpenAI emissions are estimated based on published research on cloud API carbon intensity
    - Local emissions are measured in real-time using the CodeCarbon library, which monitors your device's power usage
    
    The carbon footprint metrics should be considered approximations, as exact measurements depend on many factors including server efficiency, local power grid carbon intensity, and device specifications.
    """) 